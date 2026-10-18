import os
import joblib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import ModelRunEntity, ExperimentEntity
from app.schemas.explain import ShapSummaryResponse
from app.ml.explainer import compute_shap_explanations
from app.ml.preprocessor import PreprocessingPipelineWrapper
from app.core.config import settings

router = APIRouter()

@router.get("/{model_run_id}", response_model=ShapSummaryResponse)
def get_model_shap(model_run_id: str, db: Session = Depends(get_db)):
    run = db.query(ModelRunEntity).filter(ModelRunEntity.id == model_run_id).first()
    if not run or not run.artifact_path or not os.path.exists(run.artifact_path):
        raise HTTPException(status_code=404, detail="Model artifact not found.")

    exp = db.query(ExperimentEntity).filter(ExperimentEntity.id == run.experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    # Check cached shap
    cache_path = os.path.join(settings.ARTIFACT_DIR, f"{model_run_id}_shap.joblib")
    if os.path.exists(cache_path):
        cached = joblib.load(cache_path)
        return ShapSummaryResponse(
            model_id=run.id,
            model_name=run.model_name,
            **cached
        )

    pipeline_path = os.path.join(settings.ARTIFACT_DIR, f"{exp.id}_pipeline.joblib")
    data_path = os.path.join(settings.ARTIFACT_DIR, f"{exp.id}_data.npz")
    if not os.path.exists(pipeline_path) or not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Pipeline or training data artifacts not found.")

    pipeline_wrapper = PreprocessingPipelineWrapper.load(pipeline_path)
    model = joblib.load(run.artifact_path)
    npz = np.load(data_path)
    X_train = npz["X_train"]
    X_test = npz["X_test"]

    try:
        shap_res = compute_shap_explanations(
            model=model,
            X_train=X_train,
            X_test=X_test,
            feature_names=pipeline_wrapper.feature_names_out,
            model_name=run.model_name,
            problem_type=exp.problem_type
        )
        joblib.dump(shap_res, cache_path)

        return ShapSummaryResponse(
            model_id=run.id,
            model_name=run.model_name,
            **shap_res
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate SHAP: {str(e)}")
