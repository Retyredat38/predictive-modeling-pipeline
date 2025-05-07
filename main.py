import argparse
from src.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoML pipeline on a dataset.")
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the input CSV file (e.g., airline_ticket_price_data.csv)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of the target column to predict",
    )
    args = parser.parse_args()

    results = run_pipeline(args.csv, args.target)

    for model, report in results.items():
        print(f"\n--- {model} ---")
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                print(f"{label}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"{label}: {metrics}")
