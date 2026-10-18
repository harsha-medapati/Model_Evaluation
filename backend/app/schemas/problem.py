from typing import Dict, Any, List, Optional
from pydantic import BaseModel

class ClassDistributionItem(BaseModel):
    category: str
    count: int
    percentage: float

class TargetStats(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    median: float

class ProblemDetectionResponse(BaseModel):
    target_column: str
    detected_problem_type: str # "classification" or "regression"
    reason: str
    unique_values_count: int
    is_numeric: bool
    classes_count: Optional[int] = None
    class_distribution: Optional[List[ClassDistributionItem]] = None
    target_stats: Optional[TargetStats] = None
    recommended_metrics: List[str]

class ProblemOverrideRequest(BaseModel):
    target_column: str
    problem_type: str # "classification" or "regression"
