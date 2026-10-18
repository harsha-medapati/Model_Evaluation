import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, Response
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import ModelRunEntity, ExperimentEntity, DatasetEntity
from app.ml.exporter import create_model_bundle, create_comparison_csv, generate_pdf_report
from app.core.config import settings

router = APIRouter()

@router.get("/model/{model_run_id}")
def export_model_package(model_run_id: str, db: Session = Depends(get_db)):
    run = db.query(ModelRunEntity).filter(ModelRunEntity.id == model_run_id).first()
    if not run or not run.artifact_path or not os.path.exists(run.artifact_path):
        raise HTTPException(status_code=404, detail="Model artifact not found.")

    exp = db.query(ExperimentEntity).filter(ExperimentEntity.id == run.experiment_id).first()
    pipeline_path = os.path.join(settings.ARTIFACT_DIR, f"{exp.id}_pipeline.joblib")

    meta = {
        "model_id": run.id,
        "model_name": run.model_name,
        "model_type": run.model_type,
        "experiment_id": exp.id,
        "target_column": exp.target_column,
        "metrics": run.metrics,
        "created_at": run.created_at.isoformat() if run.created_at else None
    }

    zip_buffer = create_model_bundle(run.artifact_path, pipeline_path, meta)
    
    clean_model_name = run.model_name.lower().replace(" ", "_")
    filename = f"{clean_model_name}_package.zip"

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

@router.get("/csv/{experiment_id}")
def export_comparison_csv(experiment_id: str, db: Session = Depends(get_db)):
    runs = db.query(ModelRunEntity).filter(
        ModelRunEntity.experiment_id == experiment_id,
        ModelRunEntity.status == "completed"
    ).all()

    if not runs:
        raise HTTPException(status_code=404, detail="No completed models found for this experiment.")

    models_data = []
    for rank, r in enumerate(runs, start=1):
        cv = r.cv_scores or {}
        models_data.append({
            "rank": rank,
            "model_name": r.model_name,
            "model_type": r.model_type,
            "metrics": r.metrics or {},
            "training_time_sec": r.training_time_sec or 0.0,
            "cv_mean": cv.get("mean"),
            "cv_std": cv.get("std")
        })

    csv_content = create_comparison_csv(models_data)
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="model_comparison_{experiment_id[:8]}.csv"'}
    )

@router.get("/pdf/{experiment_id}")
def export_pdf_report(experiment_id: str, db: Session = Depends(get_db)):
    exp = db.query(ExperimentEntity).filter(ExperimentEntity.id == experiment_id).first()
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found.")

    ds = db.query(DatasetEntity).filter(DatasetEntity.id == exp.dataset_id).first()
    dataset_info = {
        "filename": ds.filename if ds else "dataset.csv",
        "row_count": ds.row_count if ds else 0,
        "col_count": ds.col_count if ds else 0
    }

    runs = db.query(ModelRunEntity).filter(
        ModelRunEntity.experiment_id == experiment_id,
        ModelRunEntity.status == "completed"
    ).all()

    comparison_data = []
    for rank, r in enumerate(runs, start=1):
        cv = r.cv_scores or {}
        comparison_data.append({
            "rank": rank,
            "model_name": r.model_name,
            "metrics": r.metrics or {},
            "training_time_sec": r.training_time_sec or 0.0
        })

    recommended_model = {}
    shap_summary = None
    if runs:
        best_run = runs[0]
        primary_metric = "F1 Score" if exp.problem_type == "classification" else "R²"
        recommended_model = {
            "model_name": best_run.model_name,
            "primary_metric_name": primary_metric,
            "primary_metric_value": best_run.metrics.get(primary_metric, 0.0),
            "rationale": "Superior performance and fold consistency on held-out validation data."
        }

        # Check if SHAP cached for best run
        shap_cache = os.path.join(settings.ARTIFACT_DIR, f"{best_run.id}_shap.joblib")
        import joblib
        if os.path.exists(shap_cache):
            shap_summary = joblib.load(shap_cache)

    pdf_path = os.path.join(settings.REPORT_DIR, f"report_{experiment_id}.pdf")
    generate_pdf_report(
        experiment_data={
            "id": exp.id,
            "target_column": exp.target_column,
            "problem_type": exp.problem_type,
            "test_size": exp.test_size,
            "preprocessing_config": exp.preprocessing_config,
            "training_config": exp.training_config
        },
        dataset_data=dataset_info,
        comparison_data=comparison_data,
        recommended_model=recommended_model,
        shap_summary=shap_summary,
        output_filepath=pdf_path
    )

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"ML_Studio_Report_{experiment_id[:8]}.pdf"
    )
