# datapilot/tabpfn_tool.py

import os

import pandas as pd

from datapilot.config import HF_TOKEN


def run_tabpfn(df: pd.DataFrame, target: str) -> dict:
    """
    Run TabPFN on the dataset.

    The Hugging Face token is loaded from the
    project root .env file.
    """

    # Make the token available to Hugging Face libraries.
    os.environ["HF_TOKEN"] = HF_TOKEN

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        r2_score,
    )

    from tabpfn import (
        TabPFNClassifier,
        TabPFNRegressor,
    )

    # --------------------------------------------------
    # Separate features and target
    # --------------------------------------------------

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # TabPFN currently receives numeric features
    X = X.select_dtypes(
        include="number"
    )

    if X.empty:
        raise ValueError(
            "TabPFN requires at least one numeric feature."
        )

    # Fill missing numeric values
    X = X.fillna(
        X.median(
            numeric_only=True
        )
    )

    # --------------------------------------------------
    # CPU safety limit
    # --------------------------------------------------

    MAX_CPU_SAMPLES = 200

    if len(X) > MAX_CPU_SAMPLES:

        sample_size = MAX_CPU_SAMPLES

        classification = (
            not pd.api.types.is_numeric_dtype(y)
            or y.nunique() <= 10
        )

        if (
            classification
            and y.value_counts().min() >= 2
        ):

            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=sample_size,
                random_state=42,
                stratify=y,
            )

        else:

            X = X.sample(
                n=sample_size,
                random_state=42,
            )

            y = y.loc[X.index]

    # --------------------------------------------------
    # Detect problem type
    # --------------------------------------------------

    classification = (
        not pd.api.types.is_numeric_dtype(y)
        or y.nunique() <= 10
    )

    # --------------------------------------------------
    # Train / test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=(
            y
            if classification and y.nunique() <= 20
            else None
        ),
    )

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    if classification:

        model = TabPFNClassifier()

        model.fit(
            X_train,
            y_train,
        )

        pred = model.predict(
            X_test
        )

        return {
            "model": "TabPFN",
            "problem_type": "classification",
            "samples_used": int(len(X)),
            "accuracy": float(
                accuracy_score(
                    y_test,
                    pred,
                )
            ),
            "f1_weighted": float(
                f1_score(
                    y_test,
                    pred,
                    average="weighted",
                )
            ),
        }

    # --------------------------------------------------
    # Regression
    # --------------------------------------------------

    model = TabPFNRegressor()

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(
        X_test
    )

    return {
        "model": "TabPFN",
        "problem_type": "regression",
        "samples_used": int(len(X)),
        "mae": float(
            mean_absolute_error(
                y_test,
                pred,
            )
        ),
        "r2": float(
            r2_score(
                y_test,
                pred,
            )
        ),
    }