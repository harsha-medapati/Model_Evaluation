import os
import uuid
import shutil
import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import DatasetEntity
from app.schemas.dataset import DatasetUploadResponse, DatasetProfileResponse
from app.ml.profiler import profile_dataset
from app.core.config import settings

router = APIRouter()

def _read_df(filepath: str) -> pd.DataFrame:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format.")

@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File extension {ext} not allowed. Please upload CSV or Excel.")

    dataset_id = str(uuid.uuid4())
    save_path = os.path.join(settings.UPLOAD_DIR, f"{dataset_id}_{file.filename}")

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = os.path.getsize(save_path)
        if file_size > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            os.remove(save_path)
            raise HTTPException(status_code=400, detail="File size exceeds maximum allowed size.")

        df = _read_df(save_path)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Failed to parse dataset: {str(e)}")

    cols = list(df.columns)
    
    # Run preliminary profiling for target/id detection
    profile_res = profile_dataset(df)

    dataset_entity = DatasetEntity(
        id=dataset_id,
        filename=file.filename,
        file_path=save_path,
        file_size_bytes=file_size,
        row_count=len(df),
        col_count=len(cols),
        target_column=profile_res.overview.target_column,
        id_column=profile_res.overview.id_column
    )
    db.add(dataset_entity)
    db.commit()

    return DatasetUploadResponse(
        id=dataset_id,
        filename=file.filename,
        file_size_bytes=file_size,
        row_count=len(df),
        col_count=len(cols),
        columns=cols,
        suggested_target_column=profile_res.overview.target_column,
        suggested_id_column=profile_res.overview.id_column
    )

@router.get("/samples")
def list_sample_datasets():
    samples = []
    if os.path.exists(settings.SAMPLE_DIR):
        for f in os.listdir(settings.SAMPLE_DIR):
            if f.endswith((".csv", ".xlsx")):
                fpath = os.path.join(settings.SAMPLE_DIR, f)
                samples.append({
                    "filename": f,
                    "size_bytes": os.path.getsize(fpath),
                    "description": "Loan Approval Classification (5000 rows)" if "loan" in f else "Housing Price Regression (2000 rows)"
                })
    return samples

@router.post("/load-sample", response_model=DatasetUploadResponse)
def load_sample_dataset(filename: str = Query(...), db: Session = Depends(get_db)):
    source_path = os.path.join(settings.SAMPLE_DIR, filename)
    if not os.path.exists(source_path):
        raise HTTPException(status_code=404, detail="Sample dataset not found.")

    dataset_id = str(uuid.uuid4())
    dest_path = os.path.join(settings.UPLOAD_DIR, f"{dataset_id}_{filename}")
    shutil.copyfile(source_path, dest_path)

    df = _read_df(dest_path)
    profile_res = profile_dataset(df)

    dataset_entity = DatasetEntity(
        id=dataset_id,
        filename=filename,
        file_path=dest_path,
        file_size_bytes=os.path.getsize(dest_path),
        row_count=len(df),
        col_count=len(df.columns),
        target_column=profile_res.overview.target_column,
        id_column=profile_res.overview.id_column
    )
    db.add(dataset_entity)
    db.commit()

    return DatasetUploadResponse(
        id=dataset_id,
        filename=filename,
        file_size_bytes=dataset_entity.file_size_bytes,
        row_count=dataset_entity.row_count,
        col_count=dataset_entity.col_count,
        columns=list(df.columns),
        suggested_target_column=profile_res.overview.target_column,
        suggested_id_column=profile_res.overview.id_column
    )

@router.get("/{dataset_id}/profile", response_model=DatasetProfileResponse)
def get_dataset_profile(dataset_id: str, target_column: str = Query(None), id_column: str = Query(None), db: Session = Depends(get_db)):
    ds = db.query(DatasetEntity).filter(DatasetEntity.id == dataset_id).first()
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    df = _read_df(ds.file_path)
    target = target_column or ds.target_column
    id_col = id_column if id_column != "None" else None
    
    return profile_dataset(df, target_col=target, id_col=id_col)
