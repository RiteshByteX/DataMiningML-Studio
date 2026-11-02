# 🏦 Universal Data Mining & Machine Learning Pipeline Studio (Django)

An automated, dataset-agnostic full-stack **Django** web application designed to fulfill all requirements of the **Mini Project Guidelines**:
1. **Part 1:** Backend Data Mining & Machine Learning Pipeline (Cleaning, Imputation, IQR Outlier Treatment, Encoding, Scaling, Feature Selection, Multi-Model Training & Evaluation).
2. **Part 2:** Interactive Django Web Interface (File Upload, Data Summary Preview, Dynamic Visual Analytics Dashboard, and Real-Time Prediction Studio).

The application works universally for **any tabular dataset** (`.csv` or `.xlsx`) and comes pre-loaded with the **Financial Credibility & Loan Decision Dataset** (10,000 records).

---

## 🚀 Quick Setup & Run Instructions

### 1. Open Terminal / PowerShell
Navigate to the project root directory:
```powershell
cd D:\Django
```

### 2. Install Required Dependencies
Ensure you have Python 3.10+ installed. Install the requirements:
```powershell
pip install -r requirements.txt
```
*(Dependencies: `django`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`, `joblib`)*

### 3. Apply Database Migrations
Run the initial Django migrations:
```powershell
python manage.py migrate
```

### 4. Start the Django Development Server
Launch the server:
```powershell
python manage.py runserver
```

### 5. Access the Web Application
Open your web browser and navigate to:
```
http://127.0.0.1:8000/
```

---

## 📋 End-to-End User Workflow

### Step 1: Ingestion & Upload (`/`)
- **1-Click Sample:** Click **"Load Financial Credibility Dataset & Proceed"** to immediately evaluate the 10,000-record credit score dataset.
- **Custom Upload:** Or drag-and-drop any `.csv` or `.xlsx` file (e.g. Loan Approval, Churn, House Prices, Heart Disease).

### Step 2: Data Summary Preview & Visual Analytics (`/eda/`)
- **Dataset Dimensions:** View total records, attributes count, numeric/categorical split, and missing values.
- **Dynamic Visualizations:**
  - Target Distribution (Histogram / Density or Class Frequencies)
  - Correlation Matrix Heatmap
  - Outlier Boxplots (IQR Whiskers)
  - Feature vs Target Relationship (Regression trend or Category breakdown)
- **Data Inspection Tables:** Top 10 rows preview, statistical summary (`describe()`), and null percentage breakdown.
- **Pipeline Controls:** Choose target column, problem type (Regression or Classification), imputation strategy (Median/Mean/Mode), outlier treatment (IQR Capping), and feature scaling (StandardScaler/MinMaxScaler).

### Step 3: Model Training & Evaluation (`/train/`)
- Executes data cleaning, duplicate removal, missing value imputation, IQR winsorization/capping, label encoding, and correlation-based feature selection.
- Trains 3 competing models:
  - **Regression:** Random Forest Regressor, Linear Regression, Gradient Boosting Regressor.
  - **Classification:** Random Forest Classifier, Logistic Regression, Gradient Boosting Classifier.
- Compares models on test metrics ($R^2$, RMSE, MAE, MSE or Accuracy, Precision, Recall, F1).
- Shows Feature Importances and Diagnostic Plots (Actual vs Predicted / Residual Histogram or Confusion Matrix).
- Allows downloading the preprocessed CSV.

### Step 4: Real-Time Live Prediction Studio (`/predict/`)
- Automatically builds an interactive form based on the selected features.
- Categorical features appear as dropdowns with valid categories; numerical features appear as number inputs with calculated bounds and median defaults.
- Click **"Auto-Fill Typical / Median Values"** or select a quick profile (Prime Borrower, Moderate Borrower, High Risk) for 1-click testing.
- Displays live prediction banner:
  - For Credit Score: Numeric score, Credit Risk Band (Poor 🔴, Fair 🟡, Good 🟢, Excellent 🌟), and Bank Loan Approval Recommendation.
  - For other datasets: Continuous numerical prediction or class label with confidence percentage.

---

## 🏗️ Project Architecture
```
D:\Django\
│
├── datamining_web/               # Django Project Configuration
│   ├── settings.py               # Configured with apps, static, media
│   ├── urls.py                   # Root URL dispatcher
│   └── wsgi.py
│
├── analyzer/                     # Core App
│   ├── ml_engine.py              # ML & Data Mining Pipeline Engine
│   ├── views.py                  # Web View Handlers
│   ├── urls.py                   # App routing
│   └── templatetags/             # Custom Django template filters
│
├── data/
│   └── sample/                   # Bundled Sample Dataset (10,000 rows)
│       └── credit_score_data.csv
│
├── media/
│   ├── uploads/                  # Ingested datasets
│   └── artifacts/                # Serialized trained model & scaler
│
├── static/
│   └── css/                      # Custom responsive stylesheet
│       └── style.css
│
├── templates/                    # Modern Responsive HTML Templates
│   ├── base.html                 # Layout with Navbar & Flow indicators
│   └── analyzer/
│       ├── upload.html           # File upload & sample loader
│       ├── eda.html              # EDA & Visual Analytics Dashboard
│       ├── train.html            # Model training & Leaderboard
│       └── predict.html          # Real-time Live Prediction Form
│
├── requirements.txt              # Python library dependencies
└── manage.py                     # Django CLI
```
