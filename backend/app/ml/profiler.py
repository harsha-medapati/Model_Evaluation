import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from app.schemas.dataset import ColumnProfile, DatasetQuality, DatasetOverview, DatasetProfileResponse

def profile_dataset(df: pd.DataFrame, target_col: str = None, id_col: str = None) -> DatasetProfileResponse:
    total_rows = len(df)
    total_cols = len(df.columns)
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    numerical_cols = []
    categorical_cols = []
    boolean_cols = []
    datetime_cols = []
    constant_cols = []
    potential_id_cols = []
    high_cardinality_cols = []
    warnings = []

    if duplicate_rows > 0:
        warnings.append(f"Found {duplicate_rows} duplicate rows in dataset.")

    columns_profiles: List[ColumnProfile] = []

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_pct = round((missing_count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        unique_cnt = int(s.nunique(dropna=True))
        
        # Check constant
        is_const = (unique_cnt <= 1)
        if is_const and total_rows > 1:
            constant_cols.append(col)
            warnings.append(f"Column '{col}' is constant (has only {unique_cnt} unique value).")

        # Check potential ID
        is_potential_id = False
        col_lower = str(col).lower()
        if (unique_cnt == total_rows and total_rows > 10) or ("id" in col_lower or "key" in col_lower or "index" in col_lower) and unique_cnt > total_rows * 0.9:
            is_potential_id = True
            potential_id_cols.append(col)

        # Detect data type
        mean_val = None
        std_val = None
        min_val = None
        max_val = None
        cat_count = None
        top_cat = None

        if pd.api.types.is_bool_dtype(s):
            dtype_str = "bool"
            boolean_cols.append(col)
            cat_count = unique_cnt
            top_val = s.mode().iloc[0] if not s.mode().empty else None
            top_cat = str(top_val) if top_val is not None else None
        elif pd.api.types.is_datetime64_any_dtype(s):
            dtype_str = "datetime"
            datetime_cols.append(col)
        elif pd.api.types.is_numeric_dtype(s) and not is_potential_id:
            dtype_str = "float64" if pd.api.types.is_float_dtype(s) else "int64"
            numerical_cols.append(col)
            valid_s = s.dropna()
            if not valid_s.empty:
                mean_val = round(float(valid_s.mean()), 2)
                std_val = round(float(valid_s.std()), 2) if len(valid_s) > 1 else 0.0
                min_val = round(float(valid_s.min()), 2)
                max_val = round(float(valid_s.max()), 2)
        else:
            dtype_str = "object"
            categorical_cols.append(col)
            cat_count = unique_cnt
            top_val = s.mode().iloc[0] if not s.mode().empty else None
            top_cat = str(top_val) if top_val is not None else None
            if unique_cnt > 50 and not is_potential_id:
                high_cardinality_cols.append(col)
                warnings.append(f"Column '{col}' has high cardinality ({unique_cnt} distinct values).")

        if missing_pct > 20.0:
            warnings.append(f"Column '{col}' has {missing_pct}% missing values.")

        # Sample values
        sample_vals = [None if pd.isna(v) else (round(v, 2) if isinstance(v, float) else v) for v in s.dropna().head(3).tolist()]

        columns_profiles.append(ColumnProfile(
            name=col,
            data_type=dtype_str,
            missing_count=missing_count,
            missing_percentage=missing_pct,
            unique_count=unique_cnt,
            mean=mean_val,
            std=std_val,
            min=min_val,
            max=max_val,
            categories_count=cat_count,
            top_category=top_cat,
            is_constant=is_const,
            is_potential_id=is_potential_id,
            sample_values=sample_vals
        ))

    # Target & ID detection suggestions
    detected_target = target_col
    if not detected_target:
        # Common target names
        common_targets = ["target", "label", "loan_status", "status", "churn", "price", "sale_price", "class", "outcome", "y"]
        for ct in common_targets:
            for c in df.columns:
                if c.lower() == ct:
                    detected_target = c
                    break
            if detected_target:
                break
        if not detected_target:
            detected_target = df.columns[-1]

    detected_id = id_col
    if not detected_id and potential_id_cols:
        detected_id = potential_id_cols[0]

    overview = DatasetOverview(
        row_count=total_rows,
        col_count=total_cols,
        missing_values_count=total_missing,
        duplicate_rows_count=duplicate_rows,
        numerical_features_count=len(numerical_cols),
        categorical_features_count=len(categorical_cols),
        target_column=detected_target,
        id_column=detected_id
    )

    quality = DatasetQuality(
        total_missing_values=total_missing,
        duplicate_rows_count=duplicate_rows,
        constant_columns=constant_cols,
        high_cardinality_categoricals=high_cardinality_cols,
        potential_id_columns=potential_id_cols,
        numerical_columns=numerical_cols,
        categorical_columns=categorical_cols,
        boolean_columns=boolean_cols,
        datetime_columns=datetime_cols,
        warnings=warnings
    )

    # Preview top 10 rows
    preview_df = df.head(10).replace({np.nan: None})
    preview_data = preview_df.to_dict(orient="records")

    return DatasetProfileResponse(
        overview=overview,
        columns=columns_profiles,
        quality=quality,
        data_preview=preview_data,
        preview_columns=list(df.columns)
    )
