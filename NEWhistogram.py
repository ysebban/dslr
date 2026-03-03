from CsvManip import CsvManip
from maths import our_mean, our_std
import matplotlib.pyplot as plt


def plot_histogram(
        Ravenclaw: dict[str, list[float]],
        Slytherin: dict[str, list[float]],
        Gryffindor: dict[str, list[float]],
        Hufflepuff: dict[str, list[float]],
        feature_name: str,
        houses: set
        ):
    """
    Create 2 bar charts for a feature:
    1. Mean scores per house
    2. Std dev per house
    """
    
    house_data = [Ravenclaw, Slytherin, Gryffindor, Hufflepuff]
    
    # Calculate means for each house
    means = []
    for house_dict in house_data:
        val = our_mean(house_dict[feature_name])
        means.append(val if val is not None else 0)
    
    # Calculate std dev for each house
    stds = []
    for house_dict in house_data:
        val = our_std(house_dict[feature_name])
        stds.append(val if val is not None else 0)
    
    # Colors: blue, green, red, yellow
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Mean scores
    ax1.bar(houses, means, color=colors)
    ax1.set_xlabel("Houses")
    ax1.set_ylabel("Mean Score")
    ax1.set_title(f"{feature_name} - Mean by House")
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Std dev
    ax2.bar(houses, stds, color=colors)
    ax2.set_xlabel("Houses")
    ax2.set_ylabel("Std Deviation")
    ax2.set_title(f"{feature_name} - Std Dev by House")
    ax2.tick_params(axis='x', rotation=45)
    
   # plt.tight_layout()
    plt.show()
    plt.close()


def main(ac: int, av: list[str]) -> int:

    if ac != 2:
        print("Error: Wrong numbers of arguments")
        return 1

    dataframe = CsvManip.loadCsv(av[1])
    All_features = CsvManip.loadFeatures(dataframe)
    Ravenclaw_features = CsvManip.loadFeatures(dataframe, house="Ravenclaw")
    Slytherin_features = CsvManip.loadFeatures(dataframe, house="Slytherin")
    Gryffindor_features = CsvManip.loadFeatures(dataframe, house="Gryffindor")
    Hufflepuff_features = CsvManip.loadFeatures(dataframe, house="Hufflepuff")

    for feature_name in All_features.keys():
        print(f"Generating histogram for: {feature_name}")
        houses = dataframe["Hogwarts House"].unique()
        plot_histogram(
            Ravenclaw_features,
            Slytherin_features,
            Gryffindor_features,
            Hufflepuff_features,
            feature_name,
            houses
        )
    
    print("Done! Check histogram_*.png files.")
    return 0


if __name__ == '__main__':
    import sys
    main(len(sys.argv), sys.argv)
