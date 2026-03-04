from CsvManip import CsvManip
from maths import our_mean, our_std
import matplotlib.pyplot as plt


# Global state for navigation
current_index = 0
feature_names = []
house_data = []
houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
fig = None
ax1 = None
ax2 = None


def update_plot(idx):
    """Update the plots with data for feature at index idx"""
    global ax1, ax2
    
    feature_name = feature_names[idx]
    
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
    
    # Clear and plot
    ax1.clear()
    ax2.clear()
    
    ax1.bar(houses, means, color=colors)
    ax1.set_xlabel("Houses")
    ax1.set_ylabel("Mean Score")
    ax1.set_title(f"{feature_name} - Mean by House")
    ax1.tick_params(axis='x', rotation=45)
    
    ax2.bar(houses, stds, color=colors)
    ax2.set_xlabel("Houses")
    ax2.set_ylabel("Std Deviation")
    ax2.set_title(f"{feature_name} - Std Dev by House")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle(
        f"Feature {idx + 1}/{len(feature_names)}: {feature_name} "
        f"(Use arrows)"
    )


def on_key(event):
    """Handle keyboard events for navigation"""
    global current_index
    
    if event.key == 'right':
        if current_index < len(feature_names) - 1:
            current_index += 1
            update_plot(current_index)
            plt.draw()
    elif event.key == 'left':
        if current_index > 0:
            current_index -= 1
            update_plot(current_index)
            plt.draw()


def main(ac: int, av: list[str]) -> int:
    global current_index, feature_names, house_data
    global fig, ax1, ax2

    if ac != 2:
        print("Error: Wrong numbers of arguments")
        return 1

    dataframe = CsvManip.loadCsv(av[1])
    All_features = CsvManip.loadFeatures(dataframe)
    Ravenclaw_features = CsvManip.loadFeatures(
        dataframe, house="Ravenclaw"
    )
    Slytherin_features = CsvManip.loadFeatures(
        dataframe, house="Slytherin"
    )
    Gryffindor_features = CsvManip.loadFeatures(
        dataframe, house="Gryffindor"
    )
    Hufflepuff_features = CsvManip.loadFeatures(
        dataframe, house="Hufflepuff"
    )

    feature_names = list(All_features.keys())
    house_data = [
        Ravenclaw_features,
        Slytherin_features,
        Gryffindor_features,
        Hufflepuff_features
    ]

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_release_event', on_key)
    
    # Show first feature
    update_plot(0)
    
    msg = f"Showing {len(feature_names)} features."
    print(msg)
    print("Use Left/Right arrow keys to navigate. Close window to exit.")
    plt.show()
    
    return 0


if __name__ == '__main__':
    import sys
    main(len(sys.argv), sys.argv)
