import os
from django.conf import settings

ARTIFACTS_DIR = os.path.join(settings.BASE_DIR, 'media', 'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Set custom writable matplotlib config dir to avoid permission warnings and cache delay
os.environ['MPLCONFIGDIR'] = os.path.join(ARTIFACTS_DIR, '.mpl_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import json
import base64
import io
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


def load_dataset(file_path):
    """
    Universally loads CSV or Excel files.
    Returns: df (pandas DataFrame), metadata dictionary.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        # Try comma first, then auto-detect delimiter
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep=None, engine='python')
    
    # Strip whitespace from column headers
    df.columns = [str(c).strip() for c in df.columns]
    
    total_rows, total_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    missing_counts = df.isnull().sum().to_dict()
    total_missing = sum(missing_counts.values())
    
    # Determine default target: 'credit_score' if present, else the last column
    if 'credit_score' in df.columns:
        default_target = 'credit_score'
    else:
        default_target = df.columns[-1]
        
    return df, {
        'total_rows': total_rows,
        'total_cols': total_cols,
        'columns': list(df.columns),
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'total_missing': total_missing,
        'missing_counts': missing_counts,
        'default_target': default_target,
    }


def detect_problem_type(df, target_col):
    """
    Auto-detects whether the problem is 'regression' or 'classification'.
    """
    if target_col not in df.columns:
        return 'regression'
        
    target_series = df[target_col].dropna()
    num_unique = target_series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(target_series)
    
    if not is_numeric or num_unique <= 15:
        return 'classification'
    return 'regression'


def run_eda_and_charts(df, target_col, problem_type='regression'):
    """
    Generates dynamic summary statistics and Base64 encoded charts for any dataset.
    """
    charts = {}
    sns.set_theme(style="whitegrid", palette="muted")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if target_col in df.columns:
        if problem_type == 'classification':
            counts = df[target_col].value_counts().head(10)
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#0ea5e9")
            ax.set_title(f"Target Distribution: '{target_col}' (Class Frequencies)", fontsize=13, fontweight='bold', pad=12)
            ax.set_xlabel(target_col)
            ax.set_ylabel("Count")
            for i, v in enumerate(counts.values):
                ax.text(i, v + (max(counts.values)*0.01), str(v), ha='center', fontsize=9, fontweight='bold')
        else:
            sns.histplot(df[target_col].dropna(), kde=True, ax=ax, color="#3b82f6", bins=30)
            ax.set_title(f"Target Distribution: '{target_col}' (Density & Histogram)", fontsize=13, fontweight='bold', pad=12)
            ax.set_xlabel(target_col)
            ax.set_ylabel("Frequency")
    plt.tight_layout()
    charts['target_distribution'] = fig_to_base64(fig)
    plt.close(fig)

    # 2. Correlation Heatmap
    if len(numeric_cols) >= 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        # Top 15 numeric columns if dataset has many
        cols_for_corr = numeric_cols[:15]
        corr_matrix = df[cols_for_corr].corr().round(2)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, mask=mask, cmap="Blues", annot=True, fmt=".2f",
            linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax, annot_kws={"size": 8}
        )
        ax.set_title("Correlation Matrix Heatmap", fontsize=13, fontweight='bold', pad=12)
        plt.tight_layout()
        charts['correlation_heatmap'] = fig_to_base64(fig)
        plt.close(fig)
    else:
        charts['correlation_heatmap'] = None

    # 3. Boxplots for Outlier Detection (Top numeric features excluding target)
    plot_num_cols = [c for c in numeric_cols if c != target_col][:6]
    if plot_num_cols:
        cols_count = len(plot_num_cols)
        ncols = 3 if cols_count >= 3 else cols_count
        nrows = int(np.ceil(cols_count / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        if cols_count == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i, col in enumerate(plot_num_cols):
            sns.boxplot(y=df[col].dropna(), ax=axes[i], color="#60a5fa", width=0.4)
            axes[i].set_title(col, fontsize=10, fontweight='bold')
            axes[i].set_ylabel("")
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        fig.suptitle("Feature Outlier Analysis (Boxplots)", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        charts['boxplots'] = fig_to_base64(fig)
        plt.close(fig)
    else:
        charts['boxplots'] = None

    # 4. Target vs Top Feature Scatter / Trend Plot
    if target_col in numeric_cols and len(plot_num_cols) > 0:
        # Find feature with highest absolute correlation with target
        corrs = df[numeric_cols].corr()[target_col].drop(target_col, errors='ignore').abs()
        if not corrs.empty:
            top_feature = corrs.idxmax()
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.regplot(
                data=df.sample(min(len(df), 1000), random_state=42),
                x=top_feature, y=target_col,
                scatter_kws={'alpha': 0.4, 'color': '#0ea5e9', 's': 20},
                line_kws={'color': '#ef4444', 'linewidth': 2},
                ax=ax
            )
            ax.set_title(f"Relationship: '{target_col}' vs '{top_feature}'", fontsize=13, fontweight='bold', pad=12)
            plt.tight_layout()
            charts['scatter_plot'] = fig_to_base64(fig)
            plt.close(fig)
        else:
            charts['scatter_plot'] = None
    elif problem_type == 'classification' and len(categorical_cols) > 0:
        top_cat = [c for c in categorical_cols if c != target_col]
        if top_cat:
            cat_col = top_cat[0]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            sns.countplot(data=df, x=cat_col, hue=target_col, ax=ax, palette="Set2")
            ax.set_title(f"Class Breakdown by '{cat_col}'", fontsize=13, fontweight='bold', pad=12)
            plt.tight_layout()
            charts['scatter_plot'] = fig_to_base64(fig)
            plt.close(fig)
        else:
            charts['scatter_plot'] = None
    else:
        charts['scatter_plot'] = None

    return charts


def clean_and_preprocess(
    df,
    target_col,
    imputation_strategy='median_mode',
    outlier_method='iqr',
    outlier_action='cap',
    encoding_method='label',
    scaling_method='standard',
    corr_threshold=0.03
):
    """
    Executes standard data mining preprocessing:
    1. Duplicate Removal
    2. Missing Value Imputation
    3. Outlier Treatment (IQR Capping or Z-Score)
    4. Categorical Encoding (LabelEncoder)
    5. Feature Selection
    6. Feature Scaling
    """
    df_proc = df.copy()
    
    # 1. Duplicates
    initial_rows = len(df_proc)
    df_proc = df_proc.drop_duplicates().reset_index(drop=True)
    duplicates_removed = initial_rows - len(df_proc)
    
    # 2. Missing Value Imputation
    missing_before = df_proc.isnull().sum().to_dict()
    num_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_proc.select_dtypes(exclude=[np.number]).columns.tolist()
    
    imputation_log = {}
    for col in df_proc.columns:
        null_count = df_proc[col].isnull().sum()
        if null_count > 0:
            if col in num_cols:
                if imputation_strategy == 'mean':
                    val = df_proc[col].mean()
                else:  # default median
                    val = df_proc[col].median()
                df_proc[col] = df_proc[col].fillna(val)
                imputation_log[col] = f"{val:.2f} ({imputation_strategy.capitalize()})"
            else:
                mode_vals = df_proc[col].mode()
                val = mode_vals[0] if not mode_vals.empty else "Missing"
                df_proc[col] = df_proc[col].fillna(val)
                imputation_log[col] = f"'{val}' (Mode)"
                
    missing_after = df_proc.isnull().sum().to_dict()
    
    # 3. Outlier Detection & Treatment (IQR)
    outlier_report = []
    outlier_num_cols = [c for c in num_cols if c != target_col]
    
    for col in outlier_num_cols:
        series = df_proc[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_count = int(((series < lower_bound) | (series > upper_bound)).sum())
        outlier_report.append({
            'column': col,
            'outliers_count': outliers_count,
            'lower_bound': round(float(lower_bound), 2),
            'upper_bound': round(float(upper_bound), 2),
        })
        
        if outlier_action == 'cap' and outliers_count > 0:
            df_proc[col] = np.clip(df_proc[col], lower_bound, upper_bound)
            
    # 4. Categorical Encoding
    encoders = {}
    categorical_mappings = {}
    
    for col in cat_cols:
        if col == target_col:
            continue
        le = LabelEncoder()
        df_proc[col] = le.fit_transform(df_proc[col].astype(str))
        encoders[col] = le
        # Store class mapping for prediction dropdowns
        categorical_mappings[col] = [str(cls) for cls in le.classes_]
        
    # If target itself is categorical (classification), encode it too
    target_encoder = None
    if target_col in cat_cols:
        target_encoder = LabelEncoder()
        df_proc[target_col] = target_encoder.fit_transform(df_proc[target_col].astype(str))
        target_classes = [str(cls) for cls in target_encoder.classes_]
    else:
        target_classes = None

    # 5. Feature Selection
    available_features = [c for c in df_proc.columns if c != target_col]
    selected_features = available_features.copy()
    dropped_features = []
    
    # Correlation-based filter if target is numeric
    if pd.api.types.is_numeric_dtype(df_proc[target_col]):
        corr_series = df_proc.corr()[target_col].drop(target_col, errors='ignore').abs()
        passing_features = corr_series[corr_series >= corr_threshold].index.tolist()
        if len(passing_features) >= 3:
            selected_features = passing_features
            dropped_features = [f for f in available_features if f not in selected_features]

    return df_proc, {
        'duplicates_removed': duplicates_removed,
        'missing_before': missing_before,
        'missing_after': missing_after,
        'imputation_log': imputation_log,
        'outlier_report': outlier_report,
        'encoders': encoders,
        'categorical_mappings': categorical_mappings,
        'target_encoder': target_encoder,
        'target_classes': target_classes,
        'selected_features': selected_features,
        'dropped_features': dropped_features,
    }


def train_models(df_proc, target_col, selected_features, problem_type='regression', scaling_method='standard', test_size=0.2):
    """
    Trains multiple ML models, calculates evaluation metrics, generates evaluation charts,
    and saves the best model artifact.
    """
    X = df_proc[selected_features]
    y = df_proc[target_col]
    
    # Train / Test Split
    if problem_type == 'classification' and y.nunique() <= 10:
        # Stratified if classification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
    # Feature Scaling
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
        
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    best_model_name = None
    best_score = -float('inf')
    best_model_obj = None
    best_y_pred = None
    
    if problem_type == 'regression':
        models_to_train = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Linear Regression': LinearRegression(),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        for name, model in models_to_train.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            r2 = float(r2_score(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'R2_Score': round(r2, 4),
                'RMSE': round(rmse, 3),
                'MAE': round(mae, 3),
                'MSE': round(mse, 3),
            }
            
            if r2 > best_score:
                best_score = r2
                best_model_name = name
                best_model_obj = model
                best_y_pred = y_pred

    else:  # Classification
        models_to_train = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in models_to_train.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            rec = float(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            
            results[name] = {
                'Accuracy': round(acc, 4),
                'Precision': round(prec, 4),
                'Recall': round(rec, 4),
                'F1_Score': round(f1, 4),
            }
            
            if acc > best_score:
                best_score = acc
                best_model_name = name
                best_model_obj = model
                best_y_pred = y_pred

    # Generate Model Evaluation Visuals
    eval_charts = {}
    
    # 1. Feature Importance Chart (if tree model has feature_importances_)
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            x=[importances[i] for i in indices],
            y=[selected_features[i] for i in indices],
            ax=ax, color="#2563eb"
        )
        ax.set_title(f"Top Feature Importances ({best_model_name})", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Relative Importance")
        plt.tight_layout()
        eval_charts['feature_importance'] = fig_to_base64(fig)
        plt.close(fig)
    else:
        eval_charts['feature_importance'] = None

    # 2. Performance Diagnostic Chart
    if problem_type == 'regression' and best_y_pred is not None:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        
        # Actual vs Predicted
        axes[0].scatter(y_test, best_y_pred, alpha=0.35, color="#2563eb", s=18)
        min_v = min(float(y_test.min()), float(best_y_pred.min()))
        max_v = max(float(y_test.max()), float(best_y_pred.max()))
        axes[0].plot([min_v, max_v], [min_v, max_v], 'r--', lw=2, label="Ideal")
        axes[0].set_title(f"Actual vs Predicted ({best_model_name})", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Actual Values")
        axes[0].set_ylabel("Predicted Values")
        axes[0].legend()
        
        # Residuals Histogram
        residuals = y_test - best_y_pred
        sns.histplot(residuals, kde=True, ax=axes[1], color="#f59e0b", bins=25)
        axes[1].axvline(0, color='red', linestyle='--', lw=1.5)
        axes[1].set_title("Residual Error Distribution", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Residual (Actual - Predicted)")
        
        plt.tight_layout()
        eval_charts['diagnostics'] = fig_to_base64(fig)
        plt.close(fig)

    elif problem_type == 'classification' and best_y_pred is not None:
        cm = confusion_matrix(y_test, best_y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, ax=ax)
        ax.set_title(f"Confusion Matrix ({best_model_name})", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        plt.tight_layout()
        eval_charts['diagnostics'] = fig_to_base64(fig)
        plt.close(fig)
    else:
        eval_charts['diagnostics'] = None

    # Save artifact bundle
    artifact_bundle = {
        'model': best_model_obj,
        'model_name': best_model_name,
        'scaler': scaler,
        'selected_features': selected_features,
        'target_col': target_col,
        'problem_type': problem_type,
        'scaling_method': scaling_method,
    }
    joblib.dump(artifact_bundle, os.path.join(ARTIFACTS_DIR, 'current_model.joblib'))

    return {
        'results': results,
        'best_model_name': best_model_name,
        'best_score': round(best_score, 4),
        'problem_type': problem_type,
        'eval_charts': eval_charts,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_count': len(selected_features),
    }


def predict_instance(input_dict, encoders, target_encoder=None):
    """
    Makes a live real-time prediction for a user-submitted feature dictionary.
    """
    model_path = os.path.join(ARTIFACTS_DIR, 'current_model.joblib')
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found! Please train a model first.")
        
    bundle = joblib.load(model_path)
    model = bundle['model']
    scaler = bundle['scaler']
    features = bundle['selected_features']
    problem_type = bundle['problem_type']
    target_col = bundle['target_col']

    # Build single row dataframe
    row_df = pd.DataFrame([input_dict])

    # Encode categorical fields
    for col, le in encoders.items():
        if col in row_df.columns:
            val = str(row_df[col].iloc[0])
            classes_list = [str(c) for c in getattr(le, 'classes_', [])]
            if val in classes_list:
                row_df[col] = classes_list.index(val)
            else:
                # Handle unseen category safely
                row_df[col] = 0

    # Ensure all required features are present and numeric
    for feat in features:
        if feat not in row_df.columns:
            row_df[feat] = 0.0
        else:
            row_df[feat] = pd.to_numeric(row_df[feat], errors='coerce').fillna(0.0)

    # Order features correctly
    X_input = row_df[features]
    X_scaled = scaler.transform(X_input)

    raw_prediction = model.predict(X_scaled)[0]
    result = {
        'raw_prediction': raw_prediction,
        'target_col': target_col,
        'problem_type': problem_type,
        'features_used': features,
    }

    if problem_type == 'regression':
        score_val = round(float(raw_prediction), 2)
        result['predicted_value'] = score_val
        
        # If target looks like credit score (300-900 range)
        if 'credit' in target_col.lower() or (score_val >= 250 and score_val <= 950):
            if score_val < 600:
                band = "Poor"
                badge_class = "danger"
                decision = "High Risk - Loan Disapproval Recommended"
                icon = "bi-x-circle-fill text-danger"
            elif score_val < 700:
                band = "Fair"
                badge_class = "warning"
                decision = "Moderate Risk - Conditional Approval with Collateral"
                icon = "bi-exclamation-triangle-fill text-warning"
            elif score_val < 750:
                band = "Good"
                badge_class = "info"
                decision = "Low Risk - Standard Loan Approval"
                icon = "bi-check-circle-fill text-primary"
            else:
                band = "Excellent"
                badge_class = "success"
                decision = "Minimal Risk - Instant Prime Rate Approval"
                icon = "bi-patch-check-fill text-success"
            result['credit_band'] = band
            result['badge_class'] = badge_class
            result['decision'] = decision
            result['icon'] = icon
    else:
        # Classification
        pred_label = raw_prediction
        if target_encoder is not None:
            try:
                pred_label = target_encoder.inverse_transform([int(raw_prediction)])[0]
            except Exception:
                pred_label = str(raw_prediction)
        result['predicted_class'] = str(pred_label)
        
        # Probabilities if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)[0]
            max_prob = float(np.max(probs)) * 100.0
            result['confidence'] = round(max_prob, 1)

    return result


def fig_to_base64(fig):
    """Encodes a matplotlib figure to base64 string for direct HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return f"data:image/png;base64,{img_b64}"
