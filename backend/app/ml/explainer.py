import shap
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from app.schemas.explain import FeatureImportanceItem, ShapScatterPoint, ShapSummaryResponse, FeatureContribution

def compute_shap_explanations(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: List[str],
    model_name: str,
    problem_type: str = "classification"
) -> Dict[str, Any]:
    # Select background data (up to 100 samples)
    n_bg = min(100, len(X_train))
    bg_indices = np.random.choice(len(X_train), size=n_bg, replace=False) if len(X_train) > n_bg else np.arange(len(X_train))
    background_data = X_train[bg_indices]

    # Select explanation samples (up to 100 samples)
    n_samples = min(100, len(X_test))
    test_indices = np.random.choice(len(X_test), size=n_samples, replace=False) if len(X_test) > n_samples else np.arange(len(X_test))
    eval_data = X_test[test_indices]

    explainer = None
    explainer_type = "TreeExplainer"

    # Select appropriate explainer
    if any(tree_kw in model_name for tree_kw in ["Forest", "Tree", "XGBoost", "Gradient Boosting"]):
        try:
            explainer = shap.TreeExplainer(model)
            explainer_type = "TreeExplainer"
        except Exception:
            explainer = shap.Explainer(model.predict, background_data)
            explainer_type = "ModelAgnosticExplainer"
    elif any(lin_kw in model_name for lin_kw in ["Logistic", "Linear", "Ridge", "Lasso"]):
        try:
            explainer = shap.LinearExplainer(model, background_data)
            explainer_type = "LinearExplainer"
        except Exception:
            explainer = shap.Explainer(model.predict, background_data)
            explainer_type = "ModelAgnosticExplainer"
    else:
        # SVM, KNN, Naive Bayes: use model-agnostic Explainer with smaller background for speed
        bg_small = background_data[:30]
        eval_small = eval_data[:30]
        predict_fn = model.predict_proba if hasattr(model, "predict_proba") and problem_type == "classification" else model.predict
        explainer = shap.Explainer(predict_fn, bg_small)
        eval_data = eval_small
        explainer_type = "Kernel/PermutationExplainer"

    # Compute SHAP values
    shap_values_raw = explainer(eval_data)
    
    # Handle SHAP object formats
    if isinstance(shap_values_raw, shap.Explanation):
        vals = shap_values_raw.values
        base_vals = shap_values_raw.base_values
    else:
        vals = np.array(shap_values_raw)
        base_vals = getattr(explainer, "expected_value", 0.0)

    # For binary classification with shape (N, P, 2), take positive class index 1
    if vals.ndim == 3 and vals.shape[2] == 2:
        vals = vals[:, :, 1]
    elif vals.ndim == 3:
        vals = vals[:, :, 0]

    # Convert base_vals to scalar safely
    try:
        if isinstance(base_vals, (list, np.ndarray)):
            base_val_scalar = float(np.ravel(base_vals)[0])
        else:
            base_val_scalar = float(base_vals)
    except Exception:
        base_val_scalar = 0.0

    # Global feature importance (mean absolute SHAP)
    mean_abs_shap = np.mean(np.abs(vals), axis=0)
    
    # Ensure length matches features
    p_len = min(len(feature_names), len(mean_abs_shap))
    clean_feature_names = feature_names[:p_len]
    mean_abs_shap = mean_abs_shap[:p_len]

    importance_list = [
        {"feature": clean_feature_names[i], "importance": round(float(mean_abs_shap[i]), 4)}
        for i in range(p_len)
    ]
    # Sort descending
    importance_list.sort(key=lambda x: x["importance"], reverse=True)

    # SHAP beeswarm / scatter points (up to 40 samples per top 10 features)
    top_features = [item["feature"] for item in importance_list[:10]]
    scatter_points = []
    
    for feat in top_features:
        feat_idx = clean_feature_names.index(feat)
        feat_vals = eval_data[:, feat_idx]
        # Normalize feature values to [0, 1] for color scale
        f_min, f_max = np.min(feat_vals), np.max(feat_vals)
        f_norm = (feat_vals - f_min) / (f_max - f_min + 1e-8) if f_max > f_min else np.zeros_like(feat_vals)

        for s_idx in range(len(eval_data)):
            scatter_points.append({
                "feature": feat,
                "feature_value": round(float(f_norm[s_idx]), 3),
                "shap_value": round(float(vals[s_idx, feat_idx]), 4)
            })

    return {
        "explainer_type": explainer_type,
        "base_value": round(base_val_scalar, 4),
        "feature_importance": importance_list,
        "shap_bar_data": importance_list[:12],
        "scatter_samples": scatter_points,
        "feature_names": clean_feature_names
    }

def explain_single_sample(
    model,
    X_train: np.ndarray,
    x_single_transformed: np.ndarray,
    raw_feature_dict: Dict[str, Any],
    feature_names: List[str],
    model_name: str,
    problem_type: str = "classification"
) -> Dict[str, Any]:
    n_bg = min(50, len(X_train))
    bg = X_train[:n_bg]

    try:
        if any(tree_kw in model_name for tree_kw in ["Forest", "Tree", "XGBoost", "Gradient Boosting"]):
            explainer = shap.TreeExplainer(model)
        elif any(lin_kw in model_name for lin_kw in ["Logistic", "Linear", "Ridge", "Lasso"]):
            explainer = shap.LinearExplainer(model, bg)
        else:
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") and problem_type == "classification" else model.predict
            explainer = shap.Explainer(predict_fn, bg[:20])

        shap_res = explainer(x_single_transformed)
        if isinstance(shap_res, shap.Explanation):
            sv = shap_res.values[0]
            bv = shap_res.base_values[0]
        else:
            sv = np.array(shap_res)[0]
            bv = getattr(explainer, "expected_value", 0.0)

        if sv.ndim == 2 and sv.shape[1] == 2:
            sv = sv[:, 1]
            if isinstance(bv, (list, np.ndarray)) and len(bv) == 2:
                bv = bv[1]
        elif sv.ndim == 2:
            sv = sv[:, 0]
            if isinstance(bv, (list, np.ndarray)):
                bv = bv[0]

        bv_float = float(np.ravel(bv)[0]) if isinstance(bv, (list, np.ndarray)) else float(bv)
    except Exception:
        # Fallback to feature importance proxy if explainer throws
        sv = np.zeros(len(feature_names))
        bv_float = 0.0

    contributions = []
    p_len = min(len(feature_names), len(sv))
    for i in range(p_len):
        f_name = feature_names[i]
        c_val = float(sv[i])
        # Find raw value if available
        raw_val = raw_feature_dict.get(f_name, "")
        contributions.append({
            "feature": f_name,
            "value": raw_val,
            "contribution": round(c_val, 4)
        })

    # Sort by absolute contribution descending
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    return {
        "base_value": round(bv_float, 4),
        "contributions": contributions
    }
