import os
import uuid
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import DatasetEntity, ExperimentEntity
from app.schemas.problem import ProblemDetectionResponse
from app.schemas.preprocessing import PreprocessingConfigRequest, PreprocessingSummaryResponse
from app.ml.detector import detect_problem_type
from app.ml.preprocessor import PreprocessingPipelineWrapper
from app.core.config import settings

router = APIRouter()

def _read_df(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    return pd.read_csv(filepath) if ext == ".csv" else pd.read_excel(filepath)

@router.get("/detect", response_model=ProblemDetectionResponse)
def detect_problem(dataset_id: str = Query(...), target_column: str = Query(...), db: Session = Depends(get_db)):
    ds = db.query(DatasetEntity).filter(DatasetEntity.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _read_df(ds.file_path)
    try:
        return detect_problem_type(df, target_column)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/configure", response_model=PreprocessingSummaryResponse)
def configure_preprocessing(req: PreprocessingConfigRequest, db: Session = Depends(get_db)):
    ds = db.query(DatasetEntity).filter(DatasetEntity.id == req.dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _read_df(ds.file_path)
    if req.target_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{req.target_column}' not found.")

    # Detect problem type
    prob_res = detect_problem_type(df, req.target_column)
    problem_type = prob_res.detected_problem_type

    # ID column handling
    id_col = req.id_column if req.id_column and req.id_column != "None" else None

    # Instantiate wrapper
    pipeline_wrapper = PreprocessingPipelineWrapper(
        target_column=req.target_column,
        id_column=id_col,
        numerical_imputer=req.numerical_imputer,
        categorical_imputer=req.categorical_imputer,
        categorical_encoding=req.categorical_encoding,
        feature_scaling=req.feature_scaling,
        test_size=req.test_size,
        random_state=req.random_state,
        problem_type=problem_type
    )

    try:
        X_train_trans, X_test_trans, y_train, y_test = pipeline_wrapper.fit_transform_split(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fit preprocessing pipeline: {str(e)}")

    experiment_id = str(uuid.uuid4())
    
    # Save fitted pipeline wrapper artifact
    pipeline_path = os.path.join(settings.ARTIFACT_DIR, f"{experiment_id}_pipeline.joblib")
    pipeline_wrapper.save(pipeline_path)

    # Save X_train, X_test, y_train, y_test cache for training
    train_cache_path = os.path.join(settings.ARTIFACT_DIR, f"{experiment_id}_data.npz")
    import numpy as np
    np.savez_compressed(
        train_cache_path,
        X_train=X_train_trans,
        X_test=X_test_trans,
        y_train=y_train,
        y_test=y_test
    )

    exp_entity = ExperimentEntity(
        id=experiment_id,
        dataset_id=req.dataset_id,
        target_column=req.target_column,
        problem_type=problem_type,
        test_size=req.test_size,
        random_state=req.random_state,
        preprocessing_config=req.model_dump(),
        training_config={},
        status="configured"
    )
    db.add(exp_entity)
    db.commit()

    return PreprocessingSummaryResponse(
        experiment_id=experiment_id,
        numerical_features=pipeline_wrapper.numerical_features,
        categorical_features=pipeline_wrapper.categorical_features,
        target_column=req.target_column,
        id_column=id_col,
        problem_type=problem_type,
        train_samples_count=len(y_train),
        test_samples_count=len(y_test),
        features_count_after_encoding=X_train_trans.shape[1],
        configuration=req.model_dump()
    )
