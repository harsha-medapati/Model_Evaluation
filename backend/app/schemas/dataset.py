from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DatasetUploadResponse(BaseModel):
    id: str
    filename: str
    file_size_bytes: int
    row_count: int
    col_count: int
    columns: List[str]
    suggested_target_column: Optional[str] = None
    suggested_id_column: Optional[str] = None

class ColumnProfile(BaseModel):
    name: str
    data_type: str
    missing_count: int
    missing_percentage: float
    unique_count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    categories_count: Optional[int] = None
    top_category: Optional[str] = None
    is_constant: bool = False
    is_potential_id: bool = False
    sample_values: List[Any] = []

class DatasetQuality(BaseModel):
    total_missing_values: int
    duplicate_rows_count: int
    constant_columns: List[str]
    high_cardinality_categoricals: List[str]
    potential_id_columns: List[str]
    numerical_columns: List[str]
    categorical_columns: List[str]
    boolean_columns: List[str]
    datetime_columns: List[str]
    warnings: List[str]

class DatasetOverview(BaseModel):
    row_count: int
    col_count: int
    missing_values_count: int
    duplicate_rows_count: int
    numerical_features_count: int
    categorical_features_count: int
    target_column: Optional[str] = None
    id_column: Optional[str] = None

class DatasetProfileResponse(BaseModel):
    overview: DatasetOverview
    columns: List[ColumnProfile]
    quality: DatasetQuality
    data_preview: List[Dict[str, Any]]
    preview_columns: List[str]
