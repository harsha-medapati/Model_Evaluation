from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelTrainRequest(BaseModel):
    experiment_id: str
    selected_models: List[str]
    cv_folds: int = Field(default=5, ge=2, le=10)
    use_hyperparameter_tuning: bool = False
    n_jobs: int = Field(default=-1)

class ModelRunSummary(BaseModel):
    id: str
    experiment_id: str
    model_name: str
    model_type: str
    status: str
    training_time_sec: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    cv_scores: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: str

class ModelComparisonItem(BaseModel):
    id: str
    model_name: str
    model_type: str
    metrics: Dict[str, Any]
    training_time_sec: float
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    rank: int

class RecommendedModelInfo(BaseModel):
    model_id: str
    model_name: str
    primary_metric_name: str
    primary_metric_value: float
    rationale: str

class ModelComparisonResponse(BaseModel):
    experiment_id: str
    problem_type: str
    models: List[ModelComparisonItem]
    recommended_model: Optional[RecommendedModelInfo] = None
    metric_keys: List[str]

class DetailedEvaluationResponse(BaseModel):
    id: str
    model_name: str
    model_type: str
    metrics: Dict[str, Any]
    training_time_sec: float
    hyperparameters: Optional[Dict[str, Any]] = None
    cv_results: Optional[Dict[str, Any]] = None
    
    # Classification specific
    confusion_matrix: Optional[Dict[str, Any]] = None # labels, matrix, normalized
    classification_report: Optional[Dict[str, Any]] = None
    roc_curve: Optional[Dict[str, Any]] = None # fpr, tpr, auc, threshold
    pr_curve: Optional[Dict[str, Any]] = None # precision, recall, avg_precision
    
    # Regression specific
    actual_vs_predicted: Optional[List[Dict[str, Any]]] = None # actual, predicted samples
    residuals: Optional[Dict[str, Any]] = None # residuals, error distribution bins
