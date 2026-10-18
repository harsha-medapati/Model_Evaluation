from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

CLASSIFICATION_MODELS = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "Support Vector Machine (SVM)",
    "K-Nearest Neighbors (KNN)",
    "Naive Bayes",
    "XGBoost Classifier"
]

REGRESSION_MODELS = [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "Decision Tree Regressor",
    "Random Forest Regressor",
    "Support Vector Regressor",
    "KNN Regressor",
    "Gradient Boosting Regressor",
    "XGBoost Regressor"
]

def get_model_instance(name: str, random_state: int = 42, n_jobs: int = -1):
    # Classification
    if name == "Logistic Regression":
        return LogisticRegression(random_state=random_state, max_iter=1000)
    elif name == "Decision Tree":
        return DecisionTreeClassifier(random_state=random_state)
    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=n_jobs)
    elif name == "Support Vector Machine (SVM)":
        return SVC(probability=True, random_state=random_state)
    elif name == "K-Nearest Neighbors (KNN)":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs)
    elif name == "Naive Bayes":
        return GaussianNB()
    elif name == "XGBoost Classifier":
        return xgb.XGBClassifier(random_state=random_state, eval_metric="logloss", n_jobs=n_jobs)

    # Regression
    elif name == "Linear Regression":
        return LinearRegression(n_jobs=n_jobs)
    elif name == "Ridge Regression":
        return Ridge(random_state=random_state)
    elif name == "Lasso Regression":
        return Lasso(random_state=random_state)
    elif name == "Decision Tree Regressor":
        return DecisionTreeRegressor(random_state=random_state)
    elif name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=n_jobs)
    elif name == "Support Vector Regressor":
        return SVR()
    elif name == "KNN Regressor":
        return KNeighborsRegressor(n_neighbors=5, n_jobs=n_jobs)
    elif name == "Gradient Boosting Regressor":
        return GradientBoostingRegressor(random_state=random_state)
    elif name == "XGBoost Regressor":
        return xgb.XGBRegressor(random_state=random_state, n_jobs=n_jobs)
    else:
        raise ValueError(f"Unknown model name: {name}")

def get_hyperparameter_grid(name: str) -> Dict[str, List[Any]]:
    # Concise grids for fast tuning
    if name == "Logistic Regression":
        return {"C": [0.1, 1.0, 10.0]}
    elif name == "Decision Tree":
        return {"max_depth": [5, 10, None], "min_samples_split": [2, 5]}
    elif name == "Random Forest":
        return {"n_estimators": [50, 100], "max_depth": [5, 12, None]}
    elif name == "Support Vector Machine (SVM)":
        return {"C": [0.5, 1.0, 5.0]}
    elif name == "K-Nearest Neighbors (KNN)":
        return {"n_neighbors": [3, 5, 7]}
    elif name == "XGBoost Classifier":
        return {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    elif name == "Ridge Regression":
        return {"alpha": [0.1, 1.0, 10.0]}
    elif name == "Lasso Regression":
        return {"alpha": [0.01, 0.1, 1.0]}
    elif name == "Random Forest Regressor":
        return {"n_estimators": [50, 100], "max_depth": [5, 12, None]}
    elif name == "Gradient Boosting Regressor":
        return {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    elif name == "XGBoost Regressor":
        return {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    return {}
