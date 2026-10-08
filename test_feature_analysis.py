import pandas as pd

from datapilot.feature_analysis import (
    analyze_numeric_features,
)


def main():

    df = pd.read_csv(
        "data/loan_default_sample.csv"
    )

    target = "default"

    result = analyze_numeric_features(
        df,
        target,
    )

    print()
    print("=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)

    print()

    print("Status:")
    print(result["status"])

    print()

    print("Target:")
    print(result["target"])

    print()

    print("Method:")
    print(result["method"])

    print()

    print("Feature evidence:")
    print()

    for feature in result["features"]:

        print(
            f"Feature: "
            f"{feature['feature']}"
        )

        print(
            f"  Correlation: "
            f"{feature['correlation']:.6f}"
        )

        print(
            f"  P-value: "
            f"{feature['p_value']:.6f}"
        )

        print(
            f"  Direction: "
            f"{feature['direction']}"
        )

        print(
            f"  Significance: "
            f"{feature['significance']}"
        )

        print(
            f"  Samples: "
            f"{feature['samples']}"
        )

        print()

    print(
        result["interpretation_note"]
    )

    print("=" * 60)


if __name__ == "__main__":
    main()