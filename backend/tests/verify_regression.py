import requests
import time

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_regression_workflow():
    print("=== TESTING REGRESSION WORKFLOW (house_prices.csv) ===")

    # 1. Load House Prices Data
    res = requests.post(f"{BASE_URL}/datasets/load-sample?filename=house_prices.csv").json()
    dataset_id = res["id"]
    print(f"[OK] House Dataset Loaded: {res['row_count']} rows, target={res['suggested_target_column']}")

    # 2. Problem Detection
    prob = requests.get(f"{BASE_URL}/preprocessing/detect?dataset_id={dataset_id}&target_column=Sale_Price").json()
    assert prob["detected_problem_type"] == "regression"
    print(f"[OK] Problem Detected: REGRESSION (Target Mean: ${prob['target_stats']['mean']:,})")

    # 3. Preprocessing
    prep_cfg = {
        "dataset_id": dataset_id,
        "target_column": "Sale_Price",
        "id_column": "Property_ID",
        "numerical_imputer": "median",
        "categorical_imputer": "most_frequent",
        "categorical_encoding": "onehot",
        "feature_scaling": "standard",
        "test_size": 0.2,
        "random_state": 42
    }
    exp = requests.post(f"{BASE_URL}/preprocessing/configure", json=prep_cfg).json()
    exp_id = exp["experiment_id"]
    print(f"[OK] Preprocessing: Train={exp['train_samples_count']}, Test={exp['test_samples_count']}")

    # 4. Train Regression Models
    models = ["Linear Regression", "Ridge Regression", "Decision Tree Regressor"]
    requests.post(f"{BASE_URL}/models/train", json={
        "experiment_id": exp_id,
        "selected_models": models,
        "cv_folds": 3,
        "use_hyperparameter_tuning": False,
        "n_jobs": 1
    })

    print("Training regression models...")
    for _ in range(30):
        time.sleep(1.5)
        st = requests.get(f"{BASE_URL}/models/status/{exp_id}").json()
        if len(st) == len(models) and all(r["status"] in ["completed", "failed"] for r in st):
            print(f"[OK] Regression models trained: {[(r['model_name'], r['status']) for r in st]}")
            break

    # 5. Compare
    comp = requests.get(f"{BASE_URL}/models/compare/{exp_id}?sort_metric=R²").json()
    rec = comp["recommended_model"]
    print(f"[OK] Champion Regressor: {rec['model_name']} (R²={rec['primary_metric_value']:.4f})")

    # 6. Detailed Evaluation
    eval_res = requests.get(f"{BASE_URL}/models/{rec['model_id']}/evaluation").json()
    print(f"[OK] Regression Metrics: R²={eval_res['metrics']['R²']}, RMSE={eval_res['metrics']['RMSE']}, MAE={eval_res['metrics']['MAE']}")
    assert len(eval_res["actual_vs_predicted"]) > 0

    print("=== REGRESSION WORKFLOW FULLY VERIFIED ===")

if __name__ == "__main__":
    test_regression_workflow()
