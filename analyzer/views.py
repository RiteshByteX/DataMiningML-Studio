import os
import shutil
import json
import pandas as pd
import numpy as np
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import HttpResponse, JsonResponse
from django.conf import settings
from . import ml_engine

DATASET_PATH = os.path.join(settings.MEDIA_ROOT, 'uploads', 'current_dataset.csv')
SAMPLE_DATASET_PATH = os.path.join(settings.BASE_DIR, 'data', 'sample', 'credit_score_data.csv')
PROCESSED_DATASET_PATH = os.path.join(settings.MEDIA_ROOT, 'uploads', 'processed_dataset.csv')


def upload_view(request):
    """File upload page with drag-and-drop & 1-click sample dataset loader."""
    if request.method == 'POST':
        uploaded_file = request.FILES.get('dataset_file')
        if not uploaded_file:
            messages.error(request, "Please select a valid .csv or .xlsx file to upload.")
            return redirect('upload')

        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext not in ['.csv', '.xlsx', '.xls']:
            messages.error(request, "Unsupported file format. Please upload a .csv or .xlsx file.")
            return redirect('upload')

        os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
        with open(DATASET_PATH, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # Clear old session data
        request.session.flush()
        request.session['dataset_name'] = uploaded_file.name
        request.session['is_sample'] = False
        messages.success(request, f"Successfully uploaded '{uploaded_file.name}'!")
        return redirect('eda')

    has_active_dataset = os.path.exists(DATASET_PATH)
    active_dataset_name = request.session.get('dataset_name', 'current_dataset.csv')
    return render(request, 'analyzer/upload.html', {
        'has_active_dataset': has_active_dataset,
        'active_dataset_name': active_dataset_name,
    })


def load_sample_view(request):
    """Loads the pre-packaged Financial Credibility & Loan Decision dataset."""
    if not os.path.exists(SAMPLE_DATASET_PATH):
        messages.error(request, "Sample dataset file not found on server.")
        return redirect('upload')

    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    shutil.copyfile(SAMPLE_DATASET_PATH, DATASET_PATH)

    request.session.flush()
    request.session['dataset_name'] = 'Financial_Credibility_credit_score_data.csv (Sample)'
    request.session['is_sample'] = True
    messages.success(request, "Loaded Financial Credibility & Loan Decision dataset (10,000 records)!")
    return redirect('eda')


def eda_view(request):
    """Exploratory Data Analysis and Visual Analytics Dashboard."""
    if not os.path.exists(DATASET_PATH):
        messages.warning(request, "No dataset loaded. Please upload a dataset or load the sample.")
        return redirect('upload')

    try:
        df, meta = ml_engine.load_dataset(DATASET_PATH)
    except Exception as e:
        messages.error(request, f"Error parsing dataset: {str(e)}")
        return redirect('upload')

    # Selected target column (from query param, session, or default)
    target_col = request.GET.get('target', request.session.get('target_col', meta['default_target']))
    if target_col not in df.columns:
        target_col = meta['default_target']
    request.session['target_col'] = target_col

    # Auto-detect problem type
    auto_problem = ml_engine.detect_problem_type(df, target_col)
    problem_type = request.GET.get('problem_type', request.session.get('problem_type', auto_problem))
    request.session['problem_type'] = problem_type

    # Generate EDA charts
    charts = ml_engine.run_eda_and_charts(df, target_col, problem_type)

    # Preview table (top 10 rows)
    preview_df = df.head(10)
    preview_html = preview_df.to_html(classes="table table-hover table-striped align-middle mb-0", index=False)

    # Statistical summary table
    summary_df = df.describe(include='all').T.reset_index().rename(columns={'index': 'Feature'})
    for col in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) and isinstance(x, (int, float, np.number)) else x)
    summary_html = summary_df.fillna('-').to_html(classes="table table-sm table-hover align-middle mb-0", index=False)

    # Missing values breakdown
    missing_data = []
    for col in df.columns:
        null_cnt = int(df[col].isnull().sum())
        missing_data.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'missing': null_cnt,
            'missing_pct': round((null_cnt / len(df)) * 100, 1)
        })

    context = {
        'dataset_name': request.session.get('dataset_name', 'Dataset'),
        'total_rows': meta['total_rows'],
        'total_cols': meta['total_cols'],
        'numeric_cols': meta['numeric_cols'],
        'categorical_cols': meta['categorical_cols'],
        'columns': meta['columns'],
        'target_col': target_col,
        'problem_type': problem_type,
        'preview_html': preview_html,
        'summary_html': summary_html,
        'missing_data': missing_data,
        'total_missing': meta['total_missing'],
        'charts': charts,
    }
    return render(request, 'analyzer/eda.html', context)


def train_view(request):
    """Executes Preprocessing and Model Training Pipeline."""
    if not os.path.exists(DATASET_PATH):
        messages.warning(request, "No dataset loaded. Please upload a dataset first.")
        return redirect('upload')

    try:
        df, meta = ml_engine.load_dataset(DATASET_PATH)
    except Exception as e:
        messages.error(request, f"Error reading dataset: {str(e)}")
        return redirect('upload')

    # Target column and problem type
    target_col = request.POST.get('target_col', request.session.get('target_col', meta['default_target']))
    problem_type = request.POST.get('problem_type', request.session.get('problem_type', ml_engine.detect_problem_type(df, target_col)))
    imputation_strategy = request.POST.get('imputation', 'median')
    outlier_action = request.POST.get('outlier_action', 'cap')
    scaling_method = request.POST.get('scaling', 'standard')
    
    # Run Preprocessing
    df_proc, prep_meta = ml_engine.clean_and_preprocess(
        df,
        target_col=target_col,
        imputation_strategy=imputation_strategy,
        outlier_action=outlier_action,
        scaling_method=scaling_method
    )

    # Save processed dataset
    df_proc.to_csv(PROCESSED_DATASET_PATH, index=False)

    # Train Models
    train_results = ml_engine.train_models(
        df_proc,
        target_col=target_col,
        selected_features=prep_meta['selected_features'],
        problem_type=problem_type,
        scaling_method=scaling_method
    )

    # Store necessary session data for real-time prediction
    request.session['target_col'] = target_col
    request.session['problem_type'] = problem_type
    request.session['selected_features'] = prep_meta['selected_features']
    request.session['categorical_mappings'] = prep_meta['categorical_mappings']
    request.session['trained_success'] = True

    # Pre-calculate feature default values and bounds for prediction form
    feature_meta = {}
    for feat in prep_meta['selected_features']:
        if feat in prep_meta['categorical_mappings']:
            feature_meta[feat] = {
                'is_categorical': True,
                'options': prep_meta['categorical_mappings'][feat],
                'default': prep_meta['categorical_mappings'][feat][0] if prep_meta['categorical_mappings'][feat] else ''
            }
        else:
            series = df_proc[feat]
            feature_meta[feat] = {
                'is_categorical': False,
                'min': round(float(series.min()), 2),
                'max': round(float(series.max()), 2),
                'default': round(float(series.median()), 2),
                'mean': round(float(series.mean()), 2),
            }
    request.session['feature_meta'] = feature_meta

    context = {
        'dataset_name': request.session.get('dataset_name', 'Dataset'),
        'target_col': target_col,
        'problem_type': problem_type,
        'prep_meta': prep_meta,
        'train_results': train_results,
        'models_comparison': train_results['results'],
        'best_model_name': train_results['best_model_name'],
        'eval_charts': train_results['eval_charts'],
        'imputation_strategy': imputation_strategy,
        'outlier_action': outlier_action,
        'scaling_method': scaling_method,
    }
    return render(request, 'analyzer/train.html', context)


def predict_view(request):
    """Interactive Real-time Prediction Studio."""
    if not request.session.get('trained_success'):
        messages.warning(request, "Please train a model first before making predictions.")
        return redirect('eda')

    selected_features = request.session.get('selected_features', [])
    feature_meta = request.session.get('feature_meta', {})
    categorical_mappings = request.session.get('categorical_mappings', {})
    target_col = request.session.get('target_col', 'Target')
    problem_type = request.session.get('problem_type', 'regression')

    prediction_result = None
    input_values = {}

    if request.method == 'POST':
        # Collect submitted inputs
        for feat in selected_features:
            val = request.POST.get(feat, '')
            input_values[feat] = val

        try:
            # Reconstruct encoders from mappings
            encoders = {}
            for col, classes in categorical_mappings.items():
                le = ml_engine.LabelEncoder()
                le.classes_ = np.array(classes)
                encoders[col] = le

            prediction_result = ml_engine.predict_instance(input_values, encoders)
            messages.success(request, "Live Prediction generated successfully!")
        except Exception as e:
            messages.error(request, f"Prediction error: {str(e)}")

    # Default values for inputs if not submitted
    if not input_values:
        for feat, meta in feature_meta.items():
            input_values[feat] = meta.get('default', '')

    context = {
        'selected_features': selected_features,
        'feature_meta': feature_meta,
        'input_values': input_values,
        'target_col': target_col,
        'problem_type': problem_type,
        'prediction_result': prediction_result,
        'dataset_name': request.session.get('dataset_name', 'Dataset'),
    }
    return render(request, 'analyzer/predict.html', context)


def download_processed_csv(request):
    """Download the preprocessed dataset as CSV."""
    if os.path.exists(PROCESSED_DATASET_PATH):
        with open(PROCESSED_DATASET_PATH, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = 'attachment; filename="preprocessed_dataset.csv"'
            return response
    messages.error(request, "Preprocessed dataset file not found.")
    return redirect('eda')
