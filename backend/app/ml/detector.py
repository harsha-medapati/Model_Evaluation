import pandas as pd
import numpy as np
from typing import Optional
from app.schemas.problem import ProblemDetectionResponse, ClassDistributionItem, TargetStats

def detect_problem_type(df: pd.DataFrame, target_col: str) -> ProblemDetectionResponse:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    s = df[target_col].dropna()
    total_samples = len(s)
    unique_vals = s.unique()
    unique_count = len(unique_vals)

    is_num = pd.api.types.is_numeric_dtype(s)
    is_bool = pd.api.types.is_bool_dtype(s)

    # Decision logic
    if is_bool:
        problem_type = "classification"
        reason = f"Classification detected because the target column is boolean with {unique_count} distinct classes."
    elif not is_num:
        problem_type = "classification"
        reason = f"Classification detected because the target column is categorical with {unique_count} unique classes."
    elif unique_count <= 10:
        problem_type = "classification"
        reason = f"Classification detected because the numerical target column has only {unique_count} discrete values."
    elif unique_count <= 20 and (unique_count / total_samples) < 0.05:
        problem_type = "classification"
        reason = f"Classification detected because the discrete numerical target has {unique_count} distinct values."
    else:
        problem_type = "regression"
        reason = f"Regression detected because the target column is continuous numeric with {unique_count} unique values."

    class_distribution = None
    target_stats = None
    classes_count = None

    if problem_type == "classification":
        classes_count = unique_count
        val_counts = s.value_counts()
        class_distribution = []
        for cat, cnt in val_counts.items():
            pct = round((cnt / total_samples) * 100, 1)
            class_distribution.append(ClassDistributionItem(
                category=str(cat),
                count=int(cnt),
                percentage=pct
            ))
        recommended_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    else:
        target_stats = TargetStats(
            mean=round(float(s.mean()), 2),
            std=round(float(s.std()), 2) if total_samples > 1 else 0.0,
            min=round(float(s.min()), 2),
            max=round(float(s.max()), 2),
            median=round(float(s.median()), 2)
        )
        recommended_metrics = ["R² Score", "RMSE", "MAE", "MSE"]

    return ProblemDetectionResponse(
        target_column=target_col,
        detected_problem_type=problem_type,
        reason=reason,
        unique_values_count=unique_count,
        is_numeric=is_num,
        classes_count=classes_count,
        class_distribution=class_distribution,
        target_stats=target_stats,
        recommended_metrics=recommended_metrics
    )
