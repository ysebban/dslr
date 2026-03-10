"""
Logistic Regression Prediction Module

Predicts Hogwarts houses using DSLR class static method.
"""

import argparse
import json
import pandas as pd
from model.Training import DSLR


def main():
    """Main function to run prediction."""
    parser = argparse.ArgumentParser(description="Predict Hogwarts houses.")
    parser.add_argument("dataset", help="Path to test CSV")
    parser.add_argument(
        "--model",
        default="model_weights.json",
        help="Model file"
    )
    parser.add_argument(
        "--output",
        default="houses.csv",
        help="Output file"
    )

    args = parser.parse_args()

    # Load test data
    print("Loading: {}".format(args.dataset))
    data = pd.read_csv(args.dataset)
    print("Loaded {} samples".format(len(data)))

    # Load model from JSON
    print("Loading model: {}".format(args.model))
    with open(args.model, 'r') as f:
        model_data = json.load(f)

    classifiers = model_data['weights']
    mean_values = model_data['mean_values']
    std_values = model_data['std_values']
    feature_names = model_data['feature_names']

    # Prepare features
    X = []
    indices = []

    for idx, row in data.iterrows():
        features = []
        for col in feature_names:
            value = row[col]
            if pd.isna(value):
                feat_idx = feature_names.index(col)
                value = mean_values[feat_idx]
            features.append(float(value))
        X.append(features)
        indices.append(idx)

    # Predict using static method
    print("Predicting...")
    predictions = []

    for i, features in enumerate(X):
        house = DSLR._predict_from_features(
            features, classifiers, mean_values, std_values)
        predictions.append((indices[i], house))

    # Save
    df = pd.DataFrame(predictions, columns=['Index', 'Hogwarts House'])
    df = df.sort_values('Index')
    df.to_csv(args.output, index=False)

    print("Predictions saved to {}".format(args.output))
    print("Total: {}".format(len(predictions)))
    print("Done!")


if __name__ == "__main__":
    main()

