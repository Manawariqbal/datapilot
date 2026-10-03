import pandas as pd


def run_tabpfn(df: pd.DataFrame, target: str) -> dict:
    # Kept isolated so the rest of DataPilot does not depend on TabPFN internals.
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        r2_score,
    )
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    X = df.drop(columns=[target])
    y = df[target]

    # MVP restriction: start with datasets whose features are numeric.
    X = X.select_dtypes(include="number")

    if X.empty:
        raise ValueError("TabPFN requires at least one numeric feature.")

    X = X.fillna(X.median(numeric_only=True))

    # ---------------------------------------------------------
    # CPU safety limit
    # ---------------------------------------------------------
    # TabPFN can be expensive on CPU for larger datasets.
    # During local development, limit the experiment to 200 rows.
    MAX_CPU_SAMPLES = 200

    if len(X) > MAX_CPU_SAMPLES:
        sample_size = MAX_CPU_SAMPLES

        # Preserve class distribution for classification when possible.
        classification = (
            not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10
        )

        if classification and y.value_counts().min() >= 2:
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

    classification = (
        not pd.api.types.is_numeric_dtype(y) or y.nunique() <= 10
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if classification and y.nunique() <= 20 else None,
    )

    # ---------------------------------------------------------
    # Classification
    # ---------------------------------------------------------
    if classification:
        model = TabPFNClassifier()

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        return {
            "model": "TabPFN",
            "problem_type": "classification",
            "samples_used": int(len(X)),
            "accuracy": float(
                accuracy_score(y_test, pred)
            ),
            "f1_weighted": float(
                f1_score(
                    y_test,
                    pred,
                    average="weighted",
                )
            ),
        }

    # ---------------------------------------------------------
    # Regression
    # ---------------------------------------------------------
    model = TabPFNRegressor()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    return {
        "model": "TabPFN",
        "problem_type": "regression",
        "samples_used": int(len(X)),
        "mae": float(
            mean_absolute_error(y_test, pred)
        ),
        "r2": float(
            r2_score(y_test, pred)
        ),
    }