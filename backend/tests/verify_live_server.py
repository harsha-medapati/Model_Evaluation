import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_full_live_workflow():
    print("=== STARTING FULL LIVE WORKFLOW VALIDATION ===")

    # 1. Samples
    res = requests.get(f"{BASE_URL}/datasets/samples")
    assert res.status_code == 200, res.text
    samples = res.json()
    print(f"[OK] Available sample datasets: {[s['filename'] for s in samples]}")

    # 2. Load Loan Data (Classification)
    res = requests.post(f"{BASE_URL}/datasets/load-sample?filename=loan_data.csv")
    assert res.status_code == 200, res.text
    ds = res.json()
    dataset_id = ds["id"]
    print(f"[OK] Dataset Loaded: ID={dataset_id}, Rows={ds['row_count']}, Cols={ds['col_count']}")

    # 3. Data Profile
    res = requests.get(f"{BASE_URL}/datasets/{dataset_id}/profile?target_column=Loan_Status&id_column=Customer_ID")
    assert res.status_code == 200, res.text
    prof = res.json()
    print(f"[OK] Data Profile: Rows={prof['overview']['row_count']}, Missing={prof['overview']['missing_values_count']}, Duplicates={prof['overview']['duplicate_rows_count']}")

    # 4. Problem Detection
    res = requests.get(f"{BASE_URL}/preprocessing/detect?dataset_id={dataset_id}&target_column=Loan_Status")
    assert res.status_code == 200, res.text
    prob = res.json()
    print(f"[OK] Problem Detected: {prob['detected_problem_type'].upper()}, Classes={prob['classes_count']}, Reason: {prob['reason']}")
    assert prob["detected_problem_type"] == "classification"

    # 5. Preprocessing
    prep_cfg = {
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
    res = requests.post(f"{BASE_URL}/preprocessing/configure", json=prep_cfg)
    assert res.status_code == 200, res.text
    exp = res.json()
    exp_id = exp["experiment_id"]
    print(f"[OK] Preprocessing Fitted: Experiment ID={exp_id}, Train Samples={exp['train_samples_count']}, Test Samples={exp['test_samples_count']}, Features After Encoding={exp['features_count_after_encoding']}")

    # 6. Train Models
    models_to_train = ["Logistic Regression", "Decision Tree", "Random Forest"]
    train_payload = {
        "experiment_id": exp_id,
        "selected_models": models_to_train,
        "cv_folds": 3,
        "use_hyperparameter_tuning": False,
        "n_jobs": 1
    }
    res = requests.post(f"{BASE_URL}/models/train", json=train_payload)
    assert res.status_code == 200, res.text
    print(f"[OK] Training Dispatched: {models_to_train}")

    # Wait for training to complete
    print("Waiting for training jobs to complete...")
    for _ in range(40):
        time.sleep(1.5)
        st_res = requests.get(f"{BASE_URL}/models/status/{exp_id}").json()
        all_done = len(st_res) == len(models_to_train) and all(r["status"] in ["completed", "failed"] for r in st_res)
        if all_done:
            print(f"[OK] All training runs completed: {[(r['model_name'], r['status']) for r in st_res]}")
            break
    else:
        raise TimeoutError("Training took too long!")

    # 7. Model Comparison
    res = requests.get(f"{BASE_URL}/models/compare/{exp_id}?sort_metric=F1 Score")
    assert res.status_code == 200, res.text
    comp = res.json()
    rec = comp["recommended_model"]
    print(f"[OK] Recommended Champion Model: {rec['model_name']} ({rec['primary_metric_name']}={rec['primary_metric_value']})")
    for m in comp["models"]:
        print(f"     Rank {m['rank']}: {m['model_name']} | Accuracy: {m['metrics'].get('Accuracy')} | F1: {m['metrics'].get('F1 Score')} | Train Time: {m['training_time_sec']}s")

    champion_id = rec["model_id"]

    # 8. Detailed Evaluation
    res = requests.get(f"{BASE_URL}/models/{champion_id}/evaluation")
    assert res.status_code == 200, res.text
    eval_data = res.json()
    print(f"[OK] Detailed Evaluation: Confusion Matrix Labels={eval_data['confusion_matrix']['labels']}, Matrix={eval_data['confusion_matrix']['matrix']}")
    print(f"     ROC-AUC={eval_data['roc_curve']['auc']}, PR Avg Precision={eval_data['pr_curve']['avg_precision']}")

    # 9. SHAP Explainability
    print("Computing SHAP explanations for champion model...")
    res = requests.get(f"{BASE_URL}/explain/{champion_id}")
    assert res.status_code == 200, res.text
    shap_data = res.json()
    print(f"[OK] SHAP Explainer: {shap_data['explainer_type']}, Base Value: {shap_data['base_value']}")
    print(f"     Top 3 Features: {[f['feature'] for f in shap_data['feature_importance'][:3]]}")

    # 10. Individual Prediction
    res = requests.get(f"{BASE_URL}/predictions/{champion_id}/schema")
    assert res.status_code == 200, res.text
    schema = res.json()
    test_input = {f["name"]: f["default_value"] for f in schema["fields"]}
    
    # Custom loan test sample
    test_input["Credit_Score"] = 760
    test_input["Income"] = 85000
    test_input["Existing_Loans"] = 0
    test_input["Loan_Amount"] = 15000

    pred_res = requests.post(f"{BASE_URL}/predictions/", json={"model_id": champion_id, "features": test_input}).json()
    print(f"[OK] Prediction Result: Decision={pred_res['prediction']}, Probability={pred_res.get('probability')}")
    print(f"     Top SHAP Contributions:")
    for fc in pred_res["feature_contributions"][:4]:
        print(f"       • {fc['feature']}: {fc['contribution']:+.3f}")

    # 11. Exports
    res_model_zip = requests.get(f"{BASE_URL}/exports/model/{champion_id}")
    assert res_model_zip.status_code == 200
    print(f"[OK] Model Package Export: {len(res_model_zip.content)} bytes zip archive")

    res_csv = requests.get(f"{BASE_URL}/exports/csv/{exp_id}")
    assert res_csv.status_code == 200
    print(f"[OK] Comparison CSV Export: {len(res_csv.content)} bytes CSV")

    res_pdf = requests.get(f"{BASE_URL}/exports/pdf/{exp_id}")
    assert res_pdf.status_code == 200
    print(f"[OK] PDF Report Export: {len(res_pdf.content)} bytes PDF report")

    print("\n=== ALL WORKFLOW STAGES VERIFIED LIVE ON RUNNING SERVER ===")

if __name__ == "__main__":
    test_full_live_workflow()
