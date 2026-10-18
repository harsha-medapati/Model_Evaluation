from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float

class ShapScatterPoint(BaseModel):
    feature: str
    feature_value: float # normalized or raw
    shap_value: float

class ShapSummaryResponse(BaseModel):
    model_id: str
    model_name: str
    explainer_type: str
    feature_importance: List[FeatureImportanceItem]
    shap_bar_data: List[FeatureImportanceItem] # mean(|SHAP|)
    scatter_samples: List[ShapScatterPoint]
    base_value: float
    feature_names: List[str]

class FeatureContribution(BaseModel):
    feature: str
    value: Any
    contribution: float # positive or negative SHAP value

class ShapIndividualResponse(BaseModel):
    prediction: Any
    probability: Optional[float] = None
    base_value: float
    output_value: float
    contributions: List[FeatureContribution]
