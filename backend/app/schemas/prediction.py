from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from app.schemas.explain import FeatureContribution

class FeatureInputField(BaseModel):
    name: str
    data_type: str # numeric, categorical, boolean
    label: str
    required: bool = True
    default_value: Optional[Any] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[str]] = None # For categoricals

class PredictionSchemaResponse(BaseModel):
    model_id: str
    model_name: str
    problem_type: str
    target_column: str
    fields: List[FeatureInputField]

class PredictionRequest(BaseModel):
    model_id: str
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    model_id: str
    prediction: Any
    probability: Optional[float] = None
    feature_contributions: List[FeatureContribution]
    base_value: Optional[float] = None
