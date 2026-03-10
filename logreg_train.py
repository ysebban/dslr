"""
Logistic Regression Training Module

Trains a logistic regression model on the given dataset.
"""

import argparse
from model.Training import DSLR
from CsvManip import CsvManip


def main():
    """Train a logistic regression model."""
    parser = argparse.ArgumentParser(description="Train logistic regression.")
    parser.add_argument(
        "dataset",
        help="Path to training CSV"
    )
    parser.add_argument(
        "--output",
        default="model_weights.json",
        help="Output file"
    )

    args = parser.parse_args()

    # Load data
    data = CsvManip.loadCsv(args.dataset)

    # Train model
    model = DSLR(data)
    model.train()

    # Save model
    model.save_model(args.output)


if __name__ == "__main__":
    main()

