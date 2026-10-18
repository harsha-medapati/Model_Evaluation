import numpy as np
from typing import Dict, Any, List, Optional
from app.ml.preprocessor import PreprocessingPipelineWrapper
from app.ml.explainer import explain_single_sample

def predict_single(
    model,
    pipeline: PreprocessingPipelineWrapper,
    features: Dict[str, Any],
    X_train_transformed: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    # 1. Transform input features
    x_trans = pipeline.transform_single_row(features)

    # 2. Model prediction
    raw_pred = model.predict(x_trans)[0]

    # Convert target if classification
    if pipeline.problem_type == "classification" and pipeline.label_encoder is not None:
        try:
            pred_label = str(pipeline.inverse_transform_target([raw_pred])[0])
        except Exception:
            pred_label = str(raw_pred)
    else:
        pred_label = round(float(raw_pred), 2)

    # 3. Probability calculation
    proba = None
    if hasattr(model, "predict_proba") and pipeline.problem_type == "classification":
        try:
            probs = model.predict_proba(x_trans)[0]
            if len(probs) == 2:
                # Probability of positive class or predicted class
                pred_idx = int(raw_pred) if int(raw_pred) < len(probs) else 1
                proba = round(float(probs[pred_idx]), 4)
            else:
                proba = round(float(np.max(probs)), 4)
        except Exception:
            proba = None

    # 4. SHAP contribution
    shap_res = explain_single_sample(
        model=model,
        X_train=X_train_transformed,
        x_single_transformed=x_trans,
        raw_feature_dict=features,
        feature_names=pipeline.feature_names_out,
        model_name=model_name,
        problem_type=pipeline.problem_type
    )

    return {
        "prediction": pred_label,
        "probability": proba,
        "feature_contributions": shap_res["contributions"][:10], # Top 10 most impactful
        "base_value": shap_res.get("base_value")
    }
