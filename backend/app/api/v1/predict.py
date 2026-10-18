import os
import uuid
import joblib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import ModelRunEntity, ExperimentEntity, PredictionHistoryEntity
from app.schemas.prediction import PredictionSchemaResponse, FeatureInputField, PredictionRequest, PredictionResponse
from app.ml.predictor import predict_single
from app.ml.preprocessor import PreprocessingPipelineWrapper
from app.core.config import settings

router = APIRouter()

@router.get("/{model_run_id}/schema", response_model=PredictionSchemaResponse)
def get_prediction_schema(model_run_id: str, db: Session = Depends(get_db)):
    run = db.query(ModelRunEntity).filter(ModelRunEntity.id == model_run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found.")

    exp = db.query(ExperimentEntity).filter(ExperimentEntity.id == run.experiment_id).first()
    pipeline_path = os.path.join(settings.ARTIFACT_DIR, f"{exp.id}_pipeline.joblib")
    if not os.path.exists(pipeline_path):
        raise HTTPException(status_code=404, detail="Pipeline artifact not found.")

    pipeline_wrapper = PreprocessingPipelineWrapper.load(pipeline_path)

    fields = []
    # Numerical fields
    for col in pipeline_wrapper.numerical_features:
        default_v = pipeline_wrapper.feature_defaults.get(col, 0.0)
        fields.append(FeatureInputField(
            name=col,
            data_type="numeric",
            label=col.replace("_", " "),
            required=True,
            default_value=default_v,
            min_value=None,
            max_value=None,
            step=1.0 if isinstance(default_v, int) else 0.1
        ))

    # Categorical fields
    for col in pipeline_wrapper.categorical_features:
        options = pipeline_wrapper.category_options.get(col, [])
        fields.append(FeatureInputField(
            name=col,
            data_type="categorical",
            label=col.replace("_", " "),
            required=True,
            default_value=options[0] if options else "",
            options=options
        ))

    return PredictionSchemaResponse(
        model_id=run.id,
        model_name=run.model_name,
        problem_type=exp.problem_type,
        target_column=exp.target_column,
        fields=fields
    )

@router.post("/", response_model=PredictionResponse)
def make_prediction(req: PredictionRequest, db: Session = Depends(get_db)):
    run = db.query(ModelRunEntity).filter(ModelRunEntity.id == req.model_id).first()
    if not run or not run.artifact_path or not os.path.exists(run.artifact_path):
        raise HTTPException(status_code=404, detail="Model not found or artifact missing.")

    exp = db.query(ExperimentEntity).filter(ExperimentEntity.id == run.experiment_id).first()
    pipeline_path = os.path.join(settings.ARTIFACT_DIR, f"{exp.id}_pipeline.joblib")
    data_path = os.path.join(settings.ARTIFACT_DIR, f"{exp.id}_data.npz")
    
    if not os.path.exists(pipeline_path) or not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail="Artifacts missing.")

    pipeline_wrapper = PreprocessingPipelineWrapper.load(pipeline_path)
    model = joblib.load(run.artifact_path)
    npz = np.load(data_path)
    X_train = npz["X_train"]

    try:
        pred_res = predict_single(
            model=model,
            pipeline=pipeline_wrapper,
            features=req.features,
            X_train_transformed=X_train,
            model_name=run.model_name
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

    # Store in history
    hist = PredictionHistoryEntity(
        id=str(uuid.uuid4()),
        model_run_id=run.id,
        input_data=req.features,
        prediction=str(pred_res["prediction"]),
        probability=pred_res.get("probability"),
        explanation=pred_res.get("feature_contributions")
    )
    db.add(hist)
    db.commit()

    return PredictionResponse(
        model_id=run.id,
        prediction=pred_res["prediction"],
        probability=pred_res.get("probability"),
        feature_contributions=pred_res["feature_contributions"],
        base_value=pred_res.get("base_value")
    )
