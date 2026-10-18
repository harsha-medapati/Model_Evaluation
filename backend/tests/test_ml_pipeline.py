import os
import pytest
import pandas as pd
import numpy as np
from app.ml.profiler import profile_dataset
from app.ml.detector import detect_problem_type
from app.ml.preprocessor import PreprocessingPipelineWrapper
from app.ml.trainer import train_single_model
from app.ml.explainer import compute_shap_explanations, explain_single_sample
from app.ml.predictor import predict_single
from app.ml.exporter import generate_pdf_report, create_model_bundle

@pytest.fixture
def loan_sample_df():
    path = "backend/data/samples/loan_data.csv"
    if not os.path.exists(path):
        from generate_samples import generate_samples
        generate_samples()
    return pd.read_csv(path)

def test_profiler(loan_sample_df):
    profile = profile_dataset(loan_sample_df, target_col="Loan_Status", id_col="Customer_ID")
    assert profile.overview.row_count > 0
    assert profile.overview.col_count == 16
    assert profile.overview.target_column == "Loan_Status"
    assert profile.overview.id_column == "Customer_ID"
    assert len(profile.columns) == 16
    assert len(profile.data_preview) <= 10

def test_detector_classification(loan_sample_df):
    res = detect_problem_type(loan_sample_df, target_col="Loan_Status")
    assert res.detected_problem_type == "classification"
    assert res.classes_count == 2
    assert len(res.class_distribution) == 2

def test_detector_regression():
    house_df = pd.read_csv("backend/data/samples/house_prices.csv")
    res = detect_problem_type(house_df, target_col="Sale_Price")
    assert res.detected_problem_type == "regression"
    assert res.target_stats is not None
    assert res.target_stats.mean > 0

def test_preprocessing_and_training_classification(loan_sample_df):
    # Take a small subset for quick test
    df_sub = loan_sample_df.head(200).copy()
    pipeline = PreprocessingPipelineWrapper(
        target_column="Loan_Status",
        id_column="Customer_ID",
        numerical_imputer="median",
        categorical_imputer="most_frequent",
        categorical_encoding="onehot",
        feature_scaling="standard",
        test_size=0.2,
        random_state=42,
        problem_type="classification"
    )

    X_train, X_test, y_train, y_test = pipeline.fit_transform_split(df_sub)
    assert X_train.shape[0] == 160
    assert X_test.shape[0] == 40
    assert len(pipeline.feature_names_out) == X_train.shape[1]

    # Train Random Forest
    rf_res = train_single_model(
        model_name="Random Forest",
        problem_type="classification",
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        cv_folds=3,
        artifact_id="test_rf"
    )

    assert "Accuracy" in rf_res["metrics"]
    assert rf_res["metrics"]["Accuracy"] > 0.5
    assert "confusion_matrix" in rf_res["evaluation_details"]
    assert rf_res["cv_scores"]["mean"] > 0

    # Test SHAP
    shap_res = compute_shap_explanations(
        model=rf_res["model"],
        X_train=X_train,
        X_test=X_test,
        feature_names=pipeline.feature_names_out,
        model_name="Random Forest",
        problem_type="classification"
    )
    assert len(shap_res["feature_importance"]) > 0
    assert len(shap_res["scatter_samples"]) > 0

    # Test Individual Prediction
    sample_input = {
        "Age": 35,
        "Income": 65000,
        "Credit_Score": 720,
        "Experience": 8,
        "Existing_Loans": 1,
        "Loan_Amount": 20000,
        "Savings_Balance": 12000,
        "Loan_Term_Months": 36,
        "Dependents": 1,
        "Education": "Bachelor",
        "Gender": "Male",
        "Employment_Type": "Salaried",
        "Home_Ownership": "Rent",
        "Has_Credit_Card": "Yes"
    }

    pred_res = predict_single(
        model=rf_res["model"],
        pipeline=pipeline,
        features=sample_input,
        X_train_transformed=X_train,
        model_name="Random Forest"
    )
    assert pred_res["prediction"] in ["Approved", "Rejected"]
    assert len(pred_res["feature_contributions"]) > 0

    # Test PDF report generation
    pdf_out = "backend/data/reports/test_report.pdf"
    generate_pdf_report(
        experiment_data={
            "id": "exp_test",
            "target_column": "Loan_Status",
            "problem_type": "classification",
            "test_size": 0.2
        },
        dataset_data={"filename": "loan_data.csv", "row_count": 5000, "col_count": 16},
        comparison_data=[{
            "rank": 1,
            "model_name": "Random Forest",
            "metrics": rf_res["metrics"],
            "training_time_sec": rf_res["training_time_sec"]
        }],
        recommended_model={
            "model_name": "Random Forest",
            "primary_metric_name": "F1 Score",
            "primary_metric_value": rf_res["metrics"]["F1 Score"],
            "rationale": "High accuracy and low variance"
        },
        shap_summary=shap_res,
        output_filepath=pdf_out
    )
    assert os.path.exists(pdf_out)
    assert os.path.getsize(pdf_out) > 1000
