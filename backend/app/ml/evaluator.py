import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error, r2_score
)

def evaluate_classification(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    
    # Probabilities
    has_proba = hasattr(model, "predict_proba")
    y_proba = None
    if has_proba:
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    unique_classes = np.unique(y_test)
    is_binary = len(unique_classes) <= 2
    avg_method = "binary" if is_binary else "weighted"

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average=avg_method, zero_division=0))
    rec = float(recall_score(y_test, y_pred, average=avg_method, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average=avg_method, zero_division=0))

    # ROC-AUC
    roc_auc_val = None
    roc_curve_data = None
    pr_curve_data = None

    if y_proba is not None:
        try:
            if is_binary:
                pos_proba = y_proba[:, 1]
                roc_auc_val = float(roc_auc_score(y_test, pos_proba))
                fpr, tpr, thresh = roc_curve(y_test, pos_proba)
                # Downsample curve points for charts if > 50
                step = max(1, len(fpr) // 50)
                roc_curve_data = {
                    "fpr": [round(float(x), 4) for x in fpr[::step]],
                    "tpr": [round(float(x), 4) for x in tpr[::step]],
                    "auc": round(roc_auc_val, 4)
                }

                p_vals, r_vals, _ = precision_recall_curve(y_test, pos_proba)
                avg_prec = float(average_precision_score(y_test, pos_proba))
                step_pr = max(1, len(p_vals) // 50)
                pr_curve_data = {
                    "precision": [round(float(x), 4) for x in p_vals[::step_pr]],
                    "recall": [round(float(x), 4) for x in r_vals[::step_pr]],
                    "avg_precision": round(avg_prec, 4)
                }
            else:
                roc_auc_val = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc_val = None

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")
    labels_list = [str(c) for c in (class_labels if class_labels is not None else unique_classes)]

    cm_data = {
        "labels": labels_list,
        "matrix": cm.tolist(),
        "normalized": [[round(float(v), 3) for v in row] for row in cm_norm.tolist()]
    }

    # Classification report
    clf_rep_dict = classification_report(
        y_test, y_pred,
        target_names=labels_list if len(labels_list) == len(unique_classes) else None,
        output_dict=True,
        zero_division=0
    )
    # Format classification report cleanly
    formatted_rep = {}
    for k, v in clf_rep_dict.items():
        if isinstance(v, dict):
            formatted_rep[k] = {
                "precision": round(float(v["precision"]), 3),
                "recall": round(float(v["recall"]), 3),
                "f1_score": round(float(v["f1-score"]), 3),
                "support": int(v["support"])
            }
        elif k == "accuracy":
            formatted_rep["accuracy"] = round(float(v), 3)

    metrics = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc_auc_val, 4) if roc_auc_val is not None else None
    }

    return {
        "metrics": metrics,
        "confusion_matrix": cm_data,
        "classification_report": formatted_rep,
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data
    }

def evaluate_regression(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_features: int
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    n = len(y_test)
    p = n_features

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    # Adjusted R2
    if n > p + 1 and (1 - r2) >= 0:
        adj_r2 = float(1 - (1 - r2) * (n - 1) / (n - p - 1))
    else:
        adj_r2 = r2

    # Residuals
    residuals = y_test - y_pred
    
    # Subsample for chart performance (max 100 points)
    sample_indices = np.random.choice(n, size=min(100, n), replace=False) if n > 100 else np.arange(n)
    actual_vs_predicted = [
        {"actual": round(float(y_test[i]), 2), "predicted": round(float(y_pred[i]), 2)}
        for i in sorted(sample_indices)
    ]

    # Histogram of residuals
    counts, bin_edges = np.histogram(residuals, bins=15)
    residual_hist = [
        {
            "bin_start": round(float(bin_edges[i]), 2),
            "bin_end": round(float(bin_edges[i+1]), 2),
            "bin_center": round(float((bin_edges[i] + bin_edges[i+1]) / 2), 2),
            "count": int(counts[i])
        }
        for i in range(len(counts))
    ]

    metrics = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R²": round(r2, 4),
        "Adjusted R²": round(adj_r2, 4)
    }

    return {
        "metrics": metrics,
        "actual_vs_predicted": actual_vs_predicted,
        "residuals": {
            "mean_residual": round(float(np.mean(residuals)), 3),
            "std_residual": round(float(np.std(residuals)), 3),
            "distribution": residual_hist
        }
    }
