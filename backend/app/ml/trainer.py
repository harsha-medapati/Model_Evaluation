import time
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, StratifiedKFold
from app.ml.registry import get_model_instance, get_hyperparameter_grid
from app.ml.evaluator import evaluate_classification, evaluate_regression
from app.ml.preprocessor import PreprocessingPipelineWrapper
from app.core.config import settings

def train_single_model(
    model_name: str,
    problem_type: str,
    pipeline: PreprocessingPipelineWrapper,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
    use_tuning: bool = False,
    n_jobs: int = -1,
    artifact_id: str = "run_1"
) -> Dict[str, Any]:
    start_time = time.time()
    
    # 1. Instantiate base model
    model = get_model_instance(model_name, random_state=pipeline.random_state, n_jobs=n_jobs)

    # 2. Cross-validation protocol
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=pipeline.random_state) \
        if problem_type == "classification" and len(np.unique(y_train)) <= 10 \
        else KFold(n_splits=cv_folds, shuffle=True, random_state=pipeline.random_state)

    scoring_metric = "f1_weighted" if problem_type == "classification" else "r2"
    
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring_metric, n_jobs=n_jobs)
        cv_results = {
            "folds": [round(float(s), 4) for s in cv_scores],
            "mean": round(float(np.mean(cv_scores)), 4),
            "std": round(float(np.std(cv_scores)), 4),
            "metric": scoring_metric
        }
    except Exception as e:
        cv_results = {
            "folds": [],
            "mean": 0.0,
            "std": 0.0,
            "error": str(e)
        }

    # 3. Optional hyperparameter tuning
    best_params = {}
    if use_tuning:
        param_grid = get_hyperparameter_grid(model_name)
        if param_grid:
            try:
                grid = GridSearchCV(model, param_grid, cv=cv_strategy, scoring=scoring_metric, n_jobs=n_jobs)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                best_params = grid.best_params_
            except Exception:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    train_duration = round(time.time() - start_time, 2)

    # 4. Evaluate on test set
    if problem_type == "classification":
        class_labels = list(pipeline.classes_) if pipeline.classes_ is not None else None
        eval_res = evaluate_classification(model, X_test, y_test, class_labels=class_labels)
    else:
        eval_res = evaluate_regression(model, X_test, y_test, n_features=X_train.shape[1])

    # 5. Save model artifact
    artifact_path = os.path.join(settings.ARTIFACT_DIR, f"{artifact_id}_model.joblib")
    joblib.dump(model, artifact_path)

    return {
        "model": model,
        "model_name": model_name,
        "model_type": problem_type,
        "training_time_sec": train_duration,
        "metrics": eval_res["metrics"],
        "cv_scores": cv_results,
        "hyperparameters": best_params,
        "artifact_path": artifact_path,
        "evaluation_details": eval_res
    }
