from typing import List, Optional
from pydantic import BaseModel, Field

class PreprocessingConfigRequest(BaseModel):
    dataset_id: str
    target_column: str
    id_column: Optional[str] = None
    numerical_imputer: str = Field(default="median", description="mean, median, most_frequent, constant")
    categorical_imputer: str = Field(default="most_frequent", description="most_frequent, constant")
    categorical_encoding: str = Field(default="onehot", description="onehot, ordinal")
    feature_scaling: str = Field(default="standard", description="standard, minmax, robust, none")
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = Field(default=42)
    stratify: bool = True
    handle_outliers: bool = False
    outlier_method: str = "none" # none, clip_iqr

class PreprocessingSummaryResponse(BaseModel):
    experiment_id: str
    numerical_features: List[str]
    categorical_features: List[str]
    target_column: str
    id_column: Optional[str] = None
    problem_type: str
    train_samples_count: int
    test_samples_count: int
    features_count_after_encoding: int
    configuration: dict
