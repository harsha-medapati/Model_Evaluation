import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

class PreprocessingPipelineWrapper:
    def __init__(
        self,
        target_column: str,
        id_column: Optional[str] = None,
        numerical_imputer: str = "median",
        categorical_imputer: str = "most_frequent",
        categorical_encoding: str = "onehot",
        feature_scaling: str = "standard",
        test_size: float = 0.2,
        random_state: int = 42,
        problem_type: str = "classification"
    ):
        self.target_column = target_column
        self.id_column = id_column
        self.numerical_imputer = numerical_imputer
        self.categorical_imputer = categorical_imputer
        self.categorical_encoding = categorical_encoding
        self.feature_scaling = feature_scaling
        self.test_size = test_size
        self.random_state = random_state
        self.problem_type = problem_type

        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names_out: List[str] = []
        self.column_transformer: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.classes_: Optional[np.ndarray] = None
        self.raw_feature_types: Dict[str, str] = {}
        self.feature_defaults: Dict[str, Any] = {}
        self.category_options: Dict[str, List[str]] = {}

    def _determine_columns(self, df: pd.DataFrame):
        feature_cols = [c for c in df.columns if c != self.target_column and c != self.id_column]
        self.numerical_features = []
        self.categorical_features = []

        for col in feature_cols:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                self.numerical_features.append(col)
                self.raw_feature_types[col] = "numeric"
                valid_s = s.dropna()
                self.feature_defaults[col] = float(valid_s.median()) if not valid_s.empty else 0.0
            else:
                self.categorical_features.append(col)
                self.raw_feature_types[col] = "categorical"
                valid_s = s.dropna().astype(str)
                cats = sorted(list(valid_s.unique()))
                self.category_options[col] = cats
                self.feature_defaults[col] = cats[0] if cats else ""

    def build_transformer(self) -> ColumnTransformer:
        transformers = []

        # Numerical pipeline
        if self.numerical_features:
            num_steps = []
            if self.numerical_imputer == "mean":
                num_steps.append(("imputer", SimpleImputer(strategy="mean")))
            elif self.numerical_imputer == "median":
                num_steps.append(("imputer", SimpleImputer(strategy="median")))
            elif self.numerical_imputer == "most_frequent":
                num_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))
            elif self.numerical_imputer == "constant":
                num_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=0.0)))
            else:
                num_steps.append(("imputer", SimpleImputer(strategy="median")))

            if self.feature_scaling == "standard":
                num_steps.append(("scaler", StandardScaler()))
            elif self.feature_scaling == "minmax":
                num_steps.append(("scaler", MinMaxScaler()))
            elif self.feature_scaling == "robust":
                num_steps.append(("scaler", RobustScaler()))
            # "none" -> no scaler

            num_pipeline = Pipeline(steps=num_steps)
            transformers.append(("num", num_pipeline, self.numerical_features))

        # Categorical pipeline
        if self.categorical_features:
            cat_steps = []
            if self.categorical_imputer == "constant":
                cat_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="Missing")))
            else:
                cat_steps.append(("imputer", SimpleImputer(strategy="most_frequent")))

            if self.categorical_encoding == "ordinal":
                cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
            else: # onehot default
                cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))

            cat_pipeline = Pipeline(steps=cat_steps)
            transformers.append(("cat", cat_pipeline, self.categorical_features))

        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit_transform_split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._determine_columns(df)
        X = df[[c for c in df.columns if c != self.target_column and c != self.id_column]]
        y = df[self.target_column].copy()

        # Handle Target for classification
        if self.problem_type == "classification":
            self.label_encoder = LabelEncoder()
            # Convert NaN in target to mode or drop
            if y.isna().any():
                y = y.fillna(y.mode().iloc[0])
            y_encoded = self.label_encoder.fit_transform(y.astype(str))
            self.classes_ = self.label_encoder.classes_
            
            # Check stratification feasibility
            class_counts = pd.Series(y_encoded).value_counts()
            can_stratify = (class_counts.min() >= 2)
            strat = y_encoded if can_stratify else None
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=self.test_size, random_state=self.random_state, stratify=strat
            )
        else:
            if y.isna().any():
                y = y.fillna(y.median())
            y_arr = y.to_numpy(dtype=float)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_arr, test_size=self.test_size, random_state=self.random_state
            )

        # Build and fit ColumnTransformer ONLY on X_train to prevent leakage
        self.column_transformer = self.build_transformer()
        X_train_trans = self.column_transformer.fit_transform(X_train)
        X_test_trans = self.column_transformer.transform(X_test)

        # Extract output feature names
        self.feature_names_out = self._get_feature_names_out()

        return X_train_trans, X_test_trans, y_train, y_test

    def _get_feature_names_out(self) -> List[str]:
        names = []
        if not self.column_transformer:
            return names
        
        for name, trans, cols in self.column_transformer.transformers_:
            if name == "num":
                names.extend(cols)
            elif name == "cat":
                encoder = trans.named_steps.get("encoder")
                if isinstance(encoder, OneHotEncoder):
                    cat_names = encoder.get_feature_names_out(cols)
                    names.extend(list(cat_names))
                elif isinstance(encoder, OrdinalEncoder):
                    names.extend(cols)
                else:
                    names.extend(cols)
        return names

    def transform_single_row(self, row_dict: Dict[str, Any]) -> np.ndarray:
        if not self.column_transformer:
            raise RuntimeError("Pipeline has not been fitted yet.")
        feature_cols = [c for c in self.numerical_features + self.categorical_features]
        # Fill missing keys with defaults
        data = {}
        for col in feature_cols:
            val = row_dict.get(col, self.feature_defaults.get(col))
            data[col] = [val]
        df_row = pd.DataFrame(data)
        return self.column_transformer.transform(df_row)

    def inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        if self.problem_type == "classification" and self.label_encoder is not None:
            return self.label_encoder.inverse_transform(np.asarray(y_pred, dtype=int))
        return y_pred

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PreprocessingPipelineWrapper":
        return joblib.load(filepath)
