# ML Model Evaluation & Explainability Studio

A production-grade Machine Learning experimentation, model evaluation, and explainability studio built with **FastAPI**, **Scikit-Learn**, **SHAP**, **XGBoost**, **ReportLab**, and **React + Vite + TypeScript + Tailwind CSS**.

The platform provides a complete end-to-end workflow: from raw dataset upload, profiling, and leakage-safe preprocessing to automatic task detection, multi-model cross-validation training, performance comparison, deep diagnostic evaluation (Confusion Matrix heatmaps, ROC/PR curves), global & local SHAP explainability, individual sample inference, and dynamic PDF report / model bundle export.

---

## Complete Studio Workflow

```
1. Upload Dataset  ──▶ 2. Data Profiling  ──▶ 3. Preprocessing  ──▶ 4. Problem Detection
        │                      │                     │                     │
   CSV / Excel           Missing values,       Imputation,           Automatic detection:
   Target selection      Types, Duplicates     Scaling, Encoding     Classification / Reg
        │                      │                     │                     │
        ▼                      ▼                     ▼                     ▼
5. Train Models    ──▶ 6. Model Compare   ──▶ 7. Detailed Eval  ──▶ 8. SHAP & Export
        │                      │                     │                     │
   Cross-validation      Metric rankings,      Confusion matrix,     SHAP beeswarm & bar,
   Asynchronous jobs     Champion Trophy       ROC / PR / Residuals  Prediction & PDF
```

---

## Key Features

1. **Dataset Upload & Ingestion**
   - Supports CSV and Excel (`.xlsx`, `.xls`) with client-side & server-side validation.
   - Smart target column detection and optional ID column exclusion.
   - Built-in sample datasets: `loan_data.csv` (Binary Classification, 5004 rows) and `house_prices.csv` (Regression, 2000 rows).

2. **Automated Data Profiling**
   - Real statistical metrics: row count, column count, total missing values, duplicate rows count.
   - Column-level profiling: data types, missing percentage, unique values, mean, std dev, min, max, top categories.
   - Data quality alerts: detects high cardinality, constant features, and potential ID columns.
   - Paginated data preview with target class badges.

3. **Reproducible Preprocessing Pipeline**
   - Built with Scikit-Learn `ColumnTransformer` and `Pipeline`.
   - **Data Leakage Safeguard**: Transformers are fitted strictly on training data split (`X_train`) only.
   - Imputation strategies: Numerical (Median, Mean, Most Frequent, Constant), Categorical (Mode, Constant).
   - Categorical Encoding: One-Hot Encoding (`handle_unknown='ignore'`), Ordinal Encoding.
   - Scaling: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, or No Scaling.
   - Split controls: configurable test size (10% to 50%), random seed, and class stratification.

4. **Automatic Problem Detection**
   - Smart heuristic analyzes discrete vs. continuous distributions and cardinalities.
   - Displays clear human-readable rationale, unique classes count, and class distribution charts.
   - Allows explicit manual override if desired.

5. **Model Registry & Training Engine**
   - **Classification Models**: Logistic Regression, Decision Tree, Random Forest, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), Naive Bayes (GaussianNB), XGBoost Classifier.
   - **Regression Models**: Linear Regression, Ridge, Lasso, Decision Tree Regressor, Random Forest Regressor, Support Vector Regressor (SVR), KNN Regressor, Gradient Boosting Regressor, XGBoost Regressor.
   - Asynchronous background training with live status tracking (`queued`, `training`, `completed`, `failed`).
   - Cross-validation scoring (`StratifiedKFold` / `KFold`) with fold metrics, mean, and standard deviation.
   - Optional hyperparameter optimization via `GridSearchCV`.

6. **Model Comparison & Recommendation**
   - Sortable multi-metric comparison table (Accuracy, Precision, Recall, F1 Score, ROC-AUC, Training Time; or R², RMSE, MAE for regression).
   - Automated Champion Recommendation badge with rationale and stability evaluation.

7. **Detailed Diagnostic Results**
   - Model switching dropdown to inspect any trained candidate.
   - **Classification**: Interactive Confusion Matrix heatmap (counts and normalized percentages) and per-class Classification Report.
   - Interactive ROC Curve with AUC score and baseline.
   - Precision-Recall Curve with Average Precision (AP).
   - Cross-Validation fold-by-fold score visualization.
   - **Regression**: Actual vs. Predicted scatter plot and Residuals distribution histogram.

8. **Real SHAP Explainability Engine**
   - Model-specific explainers: `TreeExplainer` for tree models, `LinearExplainer` for linear models, and background-sampled agnostic explainers for kernel models.
   - Global Feature Importance ranking.
   - Interactive SHAP Summary (Beeswarm) Plot with low-to-high feature value color gradient.
   - SHAP Bar Plot displaying mean absolute impact.

9. **Individual Prediction & Live SHAP Waterfall**
   - Dynamic form automatically constructed from the trained model's feature schema.
   - Generates live model decision, class probability percentage, and individual SHAP contribution impact bars.

10. **Full Export Capabilities**
    - **Download Model Package (.zip)**: Serialized model (`model.joblib`), preprocessing pipeline (`preprocessing_pipeline.joblib`), metadata (`metadata.json`), and executable `predict.py`.
    - **Download Comparison (.csv)**: Export evaluation rankings and metrics to CSV.
    - **Generate PDF Report (.pdf)**: Dynamically generated multi-page report via **ReportLab** containing dataset summaries, preprocessing configuration, comparison table, champion model metrics, and SHAP importance rankings.

---

## Tech Stack

- **Backend**: Python 3.14+, FastAPI, Uvicorn, SQLAlchemy, Pydantic V2, Scikit-Learn, XGBoost, SHAP, NumPy, Pandas, Matplotlib, ReportLab.
- **Frontend**: React 19, TypeScript, Vite, Tailwind CSS, Lucide Icons, Recharts, Axios.
- **Database**: SQLite (zero-config, persistent at `backend/data/studio.db`).

---

## Directory Structure

```
ML/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # REST endpoints (datasets, preprocessing, models, explain, predict, exports)
│   │   ├── core/            # App settings and storage directory setup
│   │   ├── db/              # SQLAlchemy database engine and ORM entities
│   │   ├── ml/              # Core ML engine (profiler, preprocessor, detector, registry, trainer, evaluator, explainer, predictor, exporter)
│   │   ├── schemas/         # Pydantic request/response schemas
│   │   └── main.py          # FastAPI application entry point
│   ├── data/
│   │   ├── uploads/         # Uploaded dataset files
│   │   ├── samples/         # Sample datasets (loan_data.csv, house_prices.csv)
│   │   ├── artifacts/       # Saved joblib pipelines and models
│   │   └── reports/         # Generated PDF reports
│   ├── tests/               # Pytest suite & live server validation scripts
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/             # API client methods
│   │   ├── components/      # Layout and 9 workflow screen components
│   │   ├── context/         # StudioContext state provider
│   │   ├── types/           # TypeScript schema definitions
│   │   ├── App.tsx          # Root application component
│   │   └── main.tsx
│   └── vite.config.ts
├── docs/                    # Architecture, API, and ML pipeline docs
├── .env.example
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.11+ (Tested and verified on Python 3.14)
- Node.js 18+ and npm

### 2. Backend Setup
```bash
# From workspace root
pip install -r backend/requirements.txt

# Start FastAPI server
$env:PYTHONPATH="backend"; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
FastAPI interactive Swagger documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## Running Automated Tests

```bash
# Run ML Engine unit test suite
$env:PYTHONPATH="backend"; python -m pytest backend/tests/test_ml_pipeline.py

# Run API endpoint integration tests
$env:PYTHONPATH="backend"; python -m pytest backend/tests/test_api.py

# Run live server end-to-end validation
python backend/tests/verify_live_server.py
python backend/tests/verify_regression.py
```

---

## License
MIT License.
