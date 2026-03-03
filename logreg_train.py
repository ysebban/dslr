import argparse
# from model.Training import Training


def main():
    """
    Train A Logistic Regression model on the given dataset.
    """
    parser = argparse.ArgumentParser(description="Train a Logistic Regression model on the given dataset.")
    parser.add_argument(
        "dataset",
        help="Path to the csv dataset"
    )
    args = parser.parse_args()
    # Load dataset
    # csv_loader = CsvManip(args.dataset)
    # data = csv_loader.load()
    print(args.dataset)
    # Train model
    # training = Training(data)
    # training.std_train()


if __name__ == "__main__":
    main()
