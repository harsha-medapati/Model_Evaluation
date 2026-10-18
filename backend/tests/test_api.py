import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_sample_datasets_list():
    res = client.get("/api/v1/datasets/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 1
    assert any("loan_data" in s["filename"] for s in samples)

def test_load_sample_and_profile():
    # 1. Load sample dataset
    res_load = client.post("/api/v1/datasets/load-sample?filename=loan_data.csv")
    assert res_load.status_code == 200
    ds_data = res_load.json()
    dataset_id = ds_data["id"]
    assert ds_data["row_count"] > 0
    assert "Loan_Status" in ds_data["columns"]

    # 2. Get profile
    res_profile = client.get(f"/api/v1/datasets/{dataset_id}/profile")
    assert res_profile.status_code == 200
    profile = res_profile.json()
    assert profile["overview"]["row_count"] > 0
    assert len(profile["columns"]) == ds_data["col_count"]

    # 3. Detect problem
    res_detect = client.get(f"/api/v1/preprocessing/detect?dataset_id={dataset_id}&target_column=Loan_Status")
    assert res_detect.status_code == 200
    det = res_detect.json()
    assert det["detected_problem_type"] == "classification"
    assert det["classes_count"] == 2

    # 4. Configure preprocessing
    prep_payload = {
        "dataset_id": dataset_id,
        "target_column": "Loan_Status",
        "id_column": "Customer_ID",
        "numerical_imputer": "median",
        "categorical_imputer": "most_frequent",
        "categorical_encoding": "onehot",
        "feature_scaling": "standard",
        "test_size": 0.2,
        "random_state": 42
    }
    res_prep = client.post("/api/v1/preprocessing/configure", json=prep_payload)
    assert res_prep.status_code == 200
    exp_summary = res_prep.json()
    exp_id = exp_summary["experiment_id"]
    assert exp_id is not None
    assert exp_summary["problem_type"] == "classification"

    # 5. Train models (e.g. Decision Tree for speed in test)
    train_payload = {
        "experiment_id": exp_id,
        "selected_models": ["Decision Tree"],
        "cv_folds": 2,
        "use_hyperparameter_tuning": False,
        "n_jobs": 1
    }
    res_train = client.post("/api/v1/models/train", json=train_payload)
    assert res_train.status_code == 200
    assert "run_ids" in res_train.json()
    run_id = res_train.json()["run_ids"][0]

    # In TestClient, BackgroundTasks are executed synchronously during the request!
    # 6. Check status
    res_status = client.get(f"/api/v1/models/status/{exp_id}")
    assert res_status.status_code == 200
    status_list = res_status.json()
    assert len(status_list) == 1
    assert status_list[0]["status"] == "completed"

    # 7. Model comparison
    res_comp = client.get(f"/api/v1/models/compare/{exp_id}")
    assert res_comp.status_code == 200
    comp = res_comp.json()
    assert len(comp["models"]) == 1
    assert comp["recommended_model"] is not None

    # 8. Detailed evaluation
    res_eval = client.get(f"/api/v1/models/{run_id}/evaluation")
    assert res_eval.status_code == 200
    eval_data = res_eval.json()
    assert "confusion_matrix" in eval_data

    # 9. Prediction schema & individual prediction
    res_schema = client.get(f"/api/v1/predictions/{run_id}/schema")
    assert res_schema.status_code == 200
    schema = res_schema.json()
    assert len(schema["fields"]) > 0

    pred_input = {f["name"]: f["default_value"] for f in schema["fields"]}
    res_pred = client.post("/api/v1/predictions/", json={"model_id": run_id, "features": pred_input})
    assert res_pred.status_code == 200
    pred_res = res_pred.json()
    assert pred_res["prediction"] is not None

    # 10. Export CSV & PDF
    res_csv = client.get(f"/api/v1/exports/csv/{exp_id}")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]

    res_pdf = client.get(f"/api/v1/exports/pdf/{exp_id}")
    assert res_pdf.status_code == 200
    assert "application/pdf" in res_pdf.headers["content-type"]
