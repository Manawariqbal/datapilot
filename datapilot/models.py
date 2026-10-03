import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

def build_preprocessor(X: pd.DataFrame):
    numeric = X.select_dtypes(include="number").columns
    categorical = X.select_dtypes(exclude="number").columns

    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

def run_baseline(df: pd.DataFrame, target: str) -> dict:
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if y.nunique() <= 20 else None
    )

    problem = "classification" if (
        not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10
    ) else "regression"

    if problem == "classification":
        model = LogisticRegression(max_iter=1000)
    else:
        model = Ridge()

    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    if problem == "classification":
        return {
            "model": "LogisticRegression",
            "problem_type": problem,
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        }

    return {
        "model": "Ridge",
        "problem_type": problem,
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }
