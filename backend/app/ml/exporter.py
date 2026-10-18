import os
import io
import json
import zipfile
import datetime
import pandas as pd
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_model_bundle(model_artifact_path: str, pipeline_artifact_path: str, metadata: Dict[str, Any]) -> io.BytesIO:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        if os.path.exists(model_artifact_path):
            zip_file.write(model_artifact_path, arcname="model.joblib")
        if os.path.exists(pipeline_artifact_path):
            zip_file.write(pipeline_artifact_path, arcname="preprocessing_pipeline.joblib")
        
        meta_json = json.dumps(metadata, indent=2)
        zip_file.writestr("metadata.json", meta_json)

        # Include sample inference python script
        inference_script = '''"""
Inference script for exported ML model package.
"""
import joblib
import pandas as pd

pipeline = joblib.load("preprocessing_pipeline.joblib")
model = joblib.load("model.joblib")

def predict(input_data: dict):
    transformed = pipeline.transform_single_row(input_data)
    pred = model.predict(transformed)
    if hasattr(pipeline, "inverse_transform_target"):
        return pipeline.inverse_transform_target(pred)[0]
    return pred[0]
'''
        zip_file.writestr("predict.py", inference_script)

    zip_buffer.seek(0)
    return zip_buffer

def create_comparison_csv(models_data: List[Dict[str, Any]]) -> str:
    rows = []
    for m in models_data:
        row = {
            "Rank": m.get("rank"),
            "Model": m.get("model_name"),
            "Type": m.get("model_type"),
            "Training Time (s)": m.get("training_time_sec"),
            "CV Mean": m.get("cv_mean"),
            "CV Std": m.get("cv_std")
        }
        for k, v in m.get("metrics", {}).items():
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)

def generate_pdf_report(
    experiment_data: Dict[str, Any],
    dataset_data: Dict[str, Any],
    comparison_data: List[Dict[str, Any]],
    recommended_model: Dict[str, Any],
    shap_summary: Dict[str, Any] = None,
    output_filepath: str = None
) -> str:
    if not output_filepath:
        output_filepath = f"backend/data/reports/report_{experiment_data.get('id', 'exp')}.pdf"

    doc = SimpleDocTemplate(
        output_filepath,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#0f172a'),
        fontName='Helvetica-Bold'
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=15
    )
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1e3a8a'),
        fontName='Helvetica-Bold',
        spaceBefore=14,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    badge_style = ParagraphStyle(
        'Badge',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#047857'),
        fontName='Helvetica-Bold'
    )

    story = []

    # Title & Header
    story.append(Paragraph("ML Model Evaluation & Explainability Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Session ID: {experiment_data.get('id')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2563eb'), spaceAfter=15))

    # 1. Dataset & Problem Summary
    story.append(Paragraph("1. Dataset & Problem Detection Summary", section_heading))
    ds_summary_data = [
        ["Dataset File:", dataset_data.get("filename", "N/A"), "Total Rows:", str(dataset_data.get("row_count", "N/A"))],
        ["Target Column:", experiment_data.get("target_column", "N/A"), "Total Columns:", str(dataset_data.get("col_count", "N/A"))],
        ["Problem Type:", experiment_data.get("problem_type", "N/A").capitalize(), "Test Split:", f"{int(experiment_data.get('test_size', 0.2) * 100)}%"]
    ]
    t1 = Table(ds_summary_data, colWidths=[110, 150, 110, 150])
    t1.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.HexColor('#1e293b')),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    # 2. Preprocessing Configuration
    story.append(Paragraph("2. Preprocessing & Validation Protocol", section_heading))
    prep_cfg = experiment_data.get("preprocessing_config", {})
    prep_table_data = [
        ["Numerical Imputer", str(prep_cfg.get("numerical_imputer", "median")).capitalize(), "Categorical Encoding", str(prep_cfg.get("categorical_encoding", "onehot")).capitalize()],
        ["Categorical Imputer", str(prep_cfg.get("categorical_imputer", "most_frequent")).capitalize(), "Feature Scaling", str(prep_cfg.get("feature_scaling", "standard")).capitalize()],
        ["Random State", str(prep_cfg.get("random_state", 42)), "CV Protocol", f"{experiment_data.get('training_config', {}).get('cv_folds', 5)} Folds"]
    ]
    t2 = Table(prep_table_data, colWidths=[120, 140, 120, 140])
    t2.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(t2)
    story.append(Spacer(1, 12))

    # 3. Model Comparison Table
    story.append(Paragraph("3. Model Performance Comparison", section_heading))
    if comparison_data:
        # Determine metric columns
        sample_metrics = comparison_data[0].get("metrics", {})
        metric_keys = list(sample_metrics.keys())
        
        headers = ["Rank", "Model"] + metric_keys + ["Train Time (s)"]
        table_rows = [headers]

        for m in comparison_data:
            row = [str(m.get("rank", "-")), m.get("model_name", "N/A")]
            for mk in metric_keys:
                v = m.get("metrics", {}).get(mk)
                if isinstance(v, float):
                    row.append(f"{v:.4f}" if v < 1 else f"{v:.2f}")
                else:
                    row.append(str(v if v is not None else "-"))
            row.append(f"{m.get('training_time_sec', 0):.2f}")
            table_rows.append(row)

        col_w = [40, 140] + [55] * len(metric_keys) + [65]
        t3 = Table(table_rows, colWidths=col_w)
        t3.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('ALIGN', (1,1), (1,-1), 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        story.append(t3)
    story.append(Spacer(1, 12))

    # 4. Recommended Model Card
    if recommended_model:
        story.append(Paragraph("4. Recommended Champion Model", section_heading))
        rec_text = f"<b>Recommended Model: {recommended_model.get('model_name')}</b><br/>" \
                   f"Criterion: Highest {recommended_model.get('primary_metric_name')} " \
                   f"({recommended_model.get('primary_metric_value')}).<br/>" \
                   f"Rationale: {recommended_model.get('rationale')}"
        story.append(Paragraph(rec_text, badge_style))
        story.append(Spacer(1, 10))

    # 5. SHAP Feature Importance
    if shap_summary and shap_summary.get("feature_importance"):
        story.append(Paragraph("5. Global Feature Importance (SHAP)", section_heading))
        shap_items = shap_summary.get("feature_importance", [])[:8]
        shap_table = [["Feature", "Mean Absolute SHAP Value", "Relative Impact"]]
        max_imp = max([item["importance"] for item in shap_items]) if shap_items else 1.0

        for it in shap_items:
            pct = (it["importance"] / max_imp * 100) if max_imp > 0 else 0
            shap_table.append([it["feature"], f"{it['importance']:.4f}", f"{pct:.1f}%"])

        t4 = Table(shap_table, colWidths=[180, 140, 140])
        t4.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2563eb')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ]))
        story.append(t4)

    # Build PDF
    doc.build(story)
    return output_filepath
