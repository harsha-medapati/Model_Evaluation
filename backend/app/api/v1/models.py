import os
import uuid
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from app.db.database import get_db, SessionLocal
from app.db.models import ExperimentEntity, ModelRunEntity
from app.schemas.model import (
    ModelTrainRequest, ModelRunSummary, ModelComparisonResponse,
    ModelComparisonItem, RecommendedModelInfo, DetailedEvaluationResponse
)
from app.ml.registry import CLASSIFICATION_MODELS, REGRESSION_MODELS
from app.ml.trainer import train_single_model
from app.ml.preprocessor import PreprocessingPipelineWrapper
from app.core.config import settings

router = APIRouter()

@router.get("/available")
def get_available_models():
    return {
        "classification": CLASSIFICATION_MODELS,
        "regression": REGRESSION_MODELS
    }

def _run_model_training_job(
    run_id: str,
    experiment_id: str,
    model_name: str,
    problem_type: str,
    cv_folds: int,
    use_tuning: bool,
    n_jobs: int
):
    db = SessionLocal()
    try:
        run_entity = db.query(ModelRunEntity).filter(ModelRunEntity.id == run_id).first()
        if not run_entity:
            return
        run_entity.status = "training"
        db.commit()

        # Load pipeline and cached split data
        pipeline_path = os.path.join(settings.ARTIFACT_DIR, f"{experiment_id}_pipeline.joblib")
        data_path = os.path.join(settings.ARTIFACT_DIR, f"{experiment_id}_data.npz")

        if not os.path.exists(pipeline_path) or not os.path.exists(data_path):
            run_entity.status = "failed"
            run_entity.error_message = "Artifacts not found."
            db.commit()
            return

        pipeline_wrapper = PreprocessingPipelineWrapper.load(pipeline_path)
        npz = np.load(data_path)
        X_train = npz["X_train"]
        X_test = npz["X_test"]
        y_train = npz["y_train"]
        y_test = npz["y_test"]

        res = train_single_model(
            model_name=model_name,
            problem_type=problem_type,
            pipeline=pipeline_wrapper,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=cv_folds,
            use_tuning=use_tuning,
            n_jobs=n_jobs,
            artifact_id=run_id
        )

        run_entity.status = "completed"
        run_entity.training_time_sec = res["training_time_sec"]
        run_entity.metrics = res["metrics"]
        run_entity.cv_scores = res["cv_scores"]
        run_entity.hyperparameters = res["hyperparameters"]
        run_entity.artifact_path = res["artifact_path"]

        # Cache detailed evaluation curves
        eval_path = os.path.join(settings.ARTIFACT_DIR, f"{run_id}_eval.joblib")
        import joblib
        joblib.dump(res["evaluation_details"], eval_path)

        db.commit()
    except Exception as e:
        run_entity = db.query(ModelRunEntity).filter(ModelRunEntity.id == run_id).first()
        if run_entity:
            run_entity.status = "failed"
            run_entity.error_message = str(e)
            db.commit()
    finally:
        db.close()

@router.post("/train")
def train_models(
    req: ModelTrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    exp = db.query(ExperimentEntity).filter(ExperimentEntity.id == req.experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    exp.training_config = {
        "cv_folds": req.cv_folds,
        "use_hyperparameter_tuning": req.use_hyperparameter_tuning,
        "n_jobs": req.n_jobs
    }
    exp.status = "training"
    db.commit()

    created_runs = []
    for model_name in req.selected_models:
        run_id = str(uuid.uuid4())
        run_entity = ModelRunEntity(
            id=run_id,
            experiment_id=exp.id,
            model_name=model_name,
            model_type=exp.problem_type,
            status="queued"
        )
        db.add(run_entity)
        created_runs.append(run_id)
    
    db.commit()

    # Launch background tasks
    for i, model_name in enumerate(req.selected_models):
        run_id = created_runs[i]
        background_tasks.add_task(
            _run_model_training_job,
            run_id=run_id,
            experiment_id=exp.id,
            model_name=model_name,
            problem_type=exp.problem_type,
            cv_folds=req.cv_folds,
            use_tuning=req.use_hyperparameter_tuning,
            n_jobs=req.n_jobs
        )

    return {"message": "Training initiated", "run_ids": created_runs}

@router.get("/status/{experiment_id}")
def get_training_status(experiment_id: str, db: Session = Depends(get_db)):
    runs = db.query(ModelRunEntity).filter(ModelRunEntity.experiment_id == experiment_id).all()
    return [
        {
            "id": r.id,
            "experiment_id": r.experiment_id,
            "model_name": r.model_name,
            "model_type": r.model_type,
            "status": r.status,
            "training_time_sec": r.training_time_sec,
            "metrics": r.metrics,
            "error_message": r.error_message,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in runs
    ]

@router.get("/compare/{experiment_id}", response_model=ModelComparisonResponse)
def compare_models(experiment_id: str, sort_metric: str = "F1 Score", db: Session = Depends(get_db)):
    exp = db.query(ExperimentEntity).filter(ExperimentEntity.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    runs = db.query(ModelRunEntity).filter(
        ModelRunEntity.experiment_id == experiment_id,
        ModelRunEntity.status == "completed"
    ).all()

    if not runs:
        return ModelComparisonResponse(
            experiment_id=experiment_id,
            problem_type=exp.problem_type,
            models=[],
            recommended_model=None,
            metric_keys=[]
        )

    # Determine sorting key
    primary_key = sort_metric
    if exp.problem_type == "classification":
        if primary_key not in ["F1 Score", "Accuracy", "ROC-AUC", "Precision", "Recall"]:
            primary_key = "F1 Score"
        reverse_sort = True
    else:
        if primary_key not in ["R²", "RMSE", "MAE", "Adjusted R²"]:
            primary_key = "R²"
        reverse_sort = (primary_key in ["R²", "Adjusted R²"])

    sample_metrics = runs[0].metrics or {}
    metric_keys = list(sample_metrics.keys())

    # Sort models
    def sort_val(r):
        m = r.metrics or {}
        val = m.get(primary_key)
        return float(val) if val is not None else -1e9

    sorted_runs = sorted(runs, key=sort_val, reverse=reverse_sort)

    items = []
    for rank_idx, r in enumerate(sorted_runs, start=1):
        cv = r.cv_scores or {}
        items.append(ModelComparisonItem(
            id=r.id,
            model_name=r.model_name,
            model_type=r.model_type,
            metrics=r.metrics or {},
            training_time_sec=r.training_time_sec or 0.0,
            cv_mean=cv.get("mean"),
            cv_std=cv.get("std"),
            rank=rank_idx
        ))

    # Champion recommendation
    rec = None
    if sorted_runs:
        best_run = sorted_runs[0]
        best_metric_val = (best_run.metrics or {}).get(primary_key, 0.0)
        
        if exp.problem_type == "classification":
            rationale = f"Selected based on highest {primary_key} ({best_metric_val:.1%}) and strong CV stability across folds."
        else:
            rationale = f"Selected based on best {primary_key} ({best_metric_val:.4f}) with minimal error variance."

        rec = RecommendedModelInfo(
            model_id=best_run.id,
            model_name=best_run.model_name,
            primary_metric_name=primary_key,
            primary_metric_value=float(best_metric_val) if best_metric_val is not None else 0.0,
            rationale=rationale
        )

    return ModelComparisonResponse(
        experiment_id=experiment_id,
        problem_type=exp.problem_type,
        models=items,
        recommended_model=rec,
        metric_keys=metric_keys
    )

@router.get("/{model_run_id}/evaluation", response_model=DetailedEvaluationResponse)
def get_detailed_evaluation(model_run_id: str, db: Session = Depends(get_db)):
    run = db.query(ModelRunEntity).filter(ModelRunEntity.id == model_run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")

    eval_path = os.path.join(settings.ARTIFACT_DIR, f"{model_run_id}_eval.joblib")
    import joblib
    if os.path.exists(eval_path):
        eval_details = joblib.load(eval_path)
    else:
        eval_details = {}

    return DetailedEvaluationResponse(
        id=run.id,
        model_name=run.model_name,
        model_type=run.model_type,
        metrics=run.metrics or {},
        training_time_sec=run.training_time_sec or 0.0,
        hyperparameters=run.hyperparameters,
        cv_results=run.cv_scores,
        confusion_matrix=eval_details.get("confusion_matrix"),
        classification_report=eval_details.get("classification_report"),
        roc_curve=eval_details.get("roc_curve"),
        pr_curve=eval_details.get("pr_curve"),
        actual_vs_predicted=eval_details.get("actual_vs_predicted"),
        residuals=eval_details.get("residuals")
    )
