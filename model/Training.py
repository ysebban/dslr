import math
import pandas as pd
import json


class DSLR:
    """
    Logistic Regression model for multi-class classification.
    Uses One-vs-Rest strategy for multiple classes.
    """

    def __init__(self, data, learning_rate=0.1, epochs=1000):
        """Initialize the model."""
        self.data = data.copy()
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Model parameters
        self.weights = None
        self.bias = None
        self.mean_values = None
        self.std_values = None
        self.feature_names = None
        self.houses = None
        self.classifiers = {}

    def train(self):
        """Start training the model."""
        return self.fit()

    def fit(self):
        """Train the model on data."""
        # Get features and labels
        X, y, feature_names = self._prepare_data()

        if X is None or len(X) == 0:
            print("Error: No valid data")
            return self

        # Store features and houses
        self.houses = list(set(y))
        self.feature_names = feature_names

        msg1 = "Training on {} samples, {} features"
        print(msg1.format(len(X), len(feature_names)))
        msg2 = "Classes: {}".format(self.houses)
        print(msg2)

        # Normalize features
        X_normalized, self.mean_values, self.std_values = self._normalize(X)

        # Add bias
        features_with_bias = self._add_bias(X_normalized)

        num_features = len(feature_names)

        # Train one classifier per house
        num_classifiers = len(self.houses)
        msg = "Training {} classifiers...".format(num_classifiers)
        print(msg)

        for house in self.houses:
            msg = "Training {}...".format(house)
            print(msg)

            # Binary labels: 1 for this house, 0 otherwise
            binary_labels = [1 if label == house else 0 for label in y]

            # Initialize weights
            weights = [0.0] * (num_features + 1)

            # Train
            weights = self._gradient_descent(features_with_bias, binary_labels, weights)

            self.classifiers[house] = weights

        print("Training complete!")

        return self

    def _prepare_data(self):
        """Extract features and labels from data."""
        house_col = "Hogwarts House"

        if house_col not in self.data.columns:
            print("Error: Hogwarts House column not found")
            return None, None, None

        # Find numeric columns
        feature_cols = []
        skip_cols = ["index", "hogwarts house", "first name",
                     "last name", "birthday", "best hand"]

        for col in self.data.columns:
            if col.lower() not in skip_cols:
                try:
                    self.data[col].astype(float)
                    feature_cols.append(col)
                except (ValueError, TypeError):
                    continue

        if not feature_cols:
            print("Error: No numeric features found")
            return None, None, None

        # Extract data
        X = []
        y = []

        for row_index, row in self.data.iterrows():
            if pd.isna(row[house_col]):
                continue

            features = []
            valid = True

            for col in feature_cols:
                value = row[col]
                if pd.isna(value):
                    valid = False
                    break
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    valid = False
                    break

            if valid:
                X.append(features)
                y.append(row[house_col])

        return X, y, feature_cols

    def _normalize(self, X):
        """Normalize features using z-score."""
        if not X:
            return [], [], []

        num_features = len(X[0])
        num_samples = len(X)

        # Calculate mean
        mean = []
        for i in range(num_features):
            vals = [X[s][i] for s in range(num_samples)]
            mean.append(sum(vals) / num_samples)

        # Calculate std
        std = []
        for i in range(num_features):
            vals = [X[s][i] for s in range(num_samples)]
            variance = sum((x - mean[i]) ** 2 for x in vals) / (num_samples - 1)
            std.append(math.sqrt(variance) if variance > 0 else 1.0)

        # Normalize
        X_normalized = []
        for s in range(num_samples):
            row = []
            for i in range(num_features):
                if std[i] == 0:
                    row.append(0)
                else:
                    val = (X[s][i] - mean[i]) / std[i]
                    row.append(val)
            X_normalized.append(row)

        return X_normalized, mean, std

    def _add_bias(self, X):
        """Add bias column to features."""
        return [[1] + row for row in X]

    def _sigmoid(self, z):
        """Compute sigmoid function."""
        if isinstance(z, list):
            return [self._sigmoid(x) for x in z]

        z = max(-500, min(500, z))
        return 1 / (1 + math.exp(-z))

    def _hypothesis(self, X, weights):
        """Compute predictions."""
        predictions = []

        for row in X:
            z = sum(row[i] * weights[i] for i in range(len(row)))
            predictions.append(self._sigmoid(z))

        return predictions

    def _compute_cost(self, y_true, y_pred, num_samples):
        """Compute loss."""
        epsilon = 1e-15

        cost = 0
        for i in range(num_samples):
            pred = max(epsilon, min(1 - epsilon, y_pred[i]))
            cost += -y_true[i] * math.log(pred) - (1 - y_true[i]) * math.log(1 - pred)

        return cost / num_samples

    def _gradient_descent(self, X, y, weights):
        """Optimize weights using gradient descent."""
        num_samples = len(y)

        for epoch in range(self.epochs):
            # Forward pass
            y_pred = self._hypothesis(X, weights)

            # Compute gradients
            gradients = [0.0] * len(weights)

            for s in range(num_samples):
                error = y_pred[s] - y[s]
                for i in range(len(weights)):
                    gradients[i] += error * X[s][i]

            # Average gradients
            for i in range(len(weights)):
                gradients[i] /= num_samples

            # Update weights
            for i in range(len(weights)):
                weights[i] -= self.learning_rate * gradients[i]

            # Print progress
            if epoch % 100 == 0:
                cost = self._compute_cost(y, y_pred, num_samples)
                msg = "Epoch {}: Cost = {:.6f}".format(epoch, cost)
                print("  " + msg)

        return weights

    def predict(self, X):
        """
        Predict house for given features.
        Returns the house name with highest probability.
        """
        if self.classifiers is None:
            print("Model not trained")
            return None

        return self._predict_from_features(X, self.classifiers,
                                           self.mean_values, self.std_values)

    @staticmethod
    def _predict_from_features(X, classifiers, mean_values, std_values):
        """Static method to predict using provided model parameters."""
        # Handle single sample
        if isinstance(X[0], (int, float)):
            X = [X]

        # Normalize
        X_normalized = []
        for row in X:
            normalized_row = []
            for i in range(len(row)):
                if std_values[i] == 0:
                    normalized_row.append(0)
                else:
                    val = (row[i] - mean_values[i]) / std_values[i]
                    normalized_row.append(val)
            X_normalized.append(normalized_row)

        # Add bias
        features_with_bias = [[1] + row for row in X_normalized]

        # Get predictions
        predictions = {}
        for house, house_weights in classifiers.items():
            z = sum(features_with_bias[0][i] * house_weights[i]
                    for i in range(len(house_weights)))
            sigmoid = 1 / (1 + math.exp(-max(-500, min(500, z))))
            predictions[house] = sigmoid

        # Return house with highest probability
        return max(predictions, key=predictions.get)

    def get_weights(self):
        """Get trained weights."""
        return self.classifiers

    def save_model(self, filepath):
        """Save model to JSON file."""
        model_data = {
            'weights': {h: w for h, w in self.classifiers.items()},
            'mean_values': self.mean_values,
            'std_values': self.std_values,
            'feature_names': self.feature_names,
            'houses': self.houses
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

        msg = "Model saved to {}".format(filepath)
        print(msg)


if __name__ == "__main__":
    """Main entry point for training."""
    import argparse
    from CsvManip import CsvManip

    parser = argparse.ArgumentParser(description="Train logistic regression.")
    parser.add_argument("dataset", help="Path to CSV")
    parser.add_argument(
        "--output",
        default="model_weights.json",
        help="Output file"
    )

    args = parser.parse_args()

    msg = "Loading: {}".format(args.dataset)
    print(msg)
    data = CsvManip.loadCsv(args.dataset)

    print("Training...")
    model = DSLR(data)
    model.train()

    model.save_model(args.output)

    print("Done!")

