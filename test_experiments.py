import pandas as pd

from datapilot.experiments import run_experiments


def main():

    df = pd.read_csv(
        "data/loan_default_sample.csv"
    )

    results = run_experiments(
        df=df,
        target="default",
        run_tabpfn_model=True,
    )

    print()
    print("=" * 60)
    print("DATAPILOT EXPERIMENT RESULTS")
    print("=" * 60)

    print("\nModels:")

    for model in results["models"]:
        print(model)

    print("\nModel Evidence:")
    print(
        results["model_evidence"]
    )

    print("\nErrors:")
    print(
        results["errors"]
    )

    print("=" * 60)


if __name__ == "__main__":
    main()