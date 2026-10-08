from datapilot.evidence import evaluate_model_comparison


def main():

    models = [
        {
            "model": "LogisticRegression",
            "problem_type": "classification",
            "accuracy": 0.77,
            "f1_weighted": 0.6786440677966101,
        },
        {
            "model": "TabPFN",
            "problem_type": "classification",
            "samples_used": 200,
            "accuracy": 0.775,
            "f1_weighted": 0.6767605633802816,
        },
    ]

    result = evaluate_model_comparison(
        models
    )

    print()
    print("=" * 60)
    print("MODEL EVIDENCE")
    print("=" * 60)

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()