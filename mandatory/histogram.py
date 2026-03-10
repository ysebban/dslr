"""
histogram.py

Interactive histogram viewer for numeric features.

The main abstraction of this file is a "histogram plot":
- A histogram plot displays one feature at a time.
- For each feature, values are grouped by house.
- Two bar charts are shown:
    - mean by house
    - std by house
- Navigation is handled by PlotNavigator.

Usage:
    python histogram.py <csv_path>
"""

from __future__ import annotations

import argparse
import math
import matplotlib.pyplot as plt

from utils.CsvManip import CsvManip
from utils.maths import Maths
from utils.PlotNavigator import PlotNavigator


HOUSES = ("Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff")


class HistogramPlot:
    """
    Histogram plot utilities for this project.

    This class prepares per-house numeric features and renders one
    feature at a time.

    The recommended usage is:
        histogram = HistogramPlot(dataframe)

        navigator = PlotNavigator(
            histogram.feature_names,
            render=histogram.render,
            make_figure=histogram.make_figure,
            title="Histogram",
        )

        navigator.show()
    """

    def __init__(self, dataframe) -> None:
        """
        Initialize a histogram plot helper.

        Args:
            dataframe: Source pandas DataFrame.

        Returns:
            A HistogramPlot instance.

        Notes:
            Numeric features are extracted once for the full dataset.
            Per-house features are also precomputed to make rendering simpler.
        """
        self.dataframe = dataframe

        all_features = CsvManip.loadFeatures(self.dataframe)
        self.feature_names = list(all_features.keys())

        self.by_house = {}
        for house_name in HOUSES:
            house_features = CsvManip.loadFeatures(
                self.dataframe,
                houses=[house_name]
            )
            self.by_house[house_name] = house_features

    def make_figure(self):
        """
        Create the matplotlib figure used for the histogram view.

        Returns:
            A matplotlib figure.

        Notes:
            Axes are created separately by make_axes().
        """
        return plt.figure(figsize=(12, 5))

    def make_axes(self, figure):
        """
        Create the axes used for the histogram view.

        Args:
            figure: Matplotlib figure where axes must be created.

        Returns:
            A tuple of axes:
                - mean_axis
                - std_axis
        """
        mean_axis, std_axis = figure.subplots(1, 2)
        return (mean_axis, std_axis)

    def render(self, feature_name: str, axes, index: int, total: int) -> str:
        """
        Render one feature for all houses.

        Args:
            feature_name: Name of the feature to display.
            axes: Tuple containing the matplotlib axes.
            index: Current feature index.
            total: Total number of features.

        Returns:
            The feature name, used by PlotNavigator in the figure title.

        Behavior:
            - Computes mean by house
            - Computes std by house
            - Draws one bar chart for means
            - Draws one bar chart for stds
        """
        mean_axis, std_axis = axes

        means = []
        stds = []

        for house in HOUSES:
            values = self.by_house[house].get(feature_name, [])
            if not values:
                means.append(math.nan)
                stds.append(math.nan)
                continue

            means.append(Maths.mean(values))
            stds.append(Maths.std(values))

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']

        mean_axis.bar(HOUSES, means, color=colors)
        mean_axis.set_title("Mean by house")
        mean_axis.set_xlabel("House")
        mean_axis.set_ylabel("Mean")
        mean_axis.tick_params(axis="x", rotation=45)

        std_axis.bar(HOUSES, stds, color=colors)
        std_axis.set_title("Std by house")
        std_axis.set_xlabel("House")
        std_axis.set_ylabel("Std")
        std_axis.tick_params(axis="x", rotation=45)

        return feature_name


def main(argv: list[str] | None = None) -> int:
    """
    Run the histogram viewer from command line arguments.

    Args:
        argv: Optional argument list. If None, uses command line arguments.

    Returns:
        Exit status code.

    Behavior:
        - Parses the CSV path
        - Loads the dataset
        - Builds the histogram helper
        - Starts the interactive navigator
    """
    parser = argparse.ArgumentParser(
        prog="histogram.py",
        description="Interactive per-house mean/std view for numeric features."
    )
    parser.add_argument("csv_path", help="Path to the CSV dataset.")
    args = parser.parse_args(argv)

    try:
        dataframe = CsvManip.loadCsv(args.csv_path)
    except Exception as exc:
        parser.error(f"Cannot load CSV: {exc!s}")
        return 1

    histogram = HistogramPlot(dataframe)
    if not histogram.feature_names:
        print("No numeric features found.")
        return 0

    navigator = PlotNavigator(
        histogram.feature_names,
        render=histogram.render,
        make_figure=histogram.make_figure,
        make_axes=histogram.make_axes,
        title="Histogram",
    )
    navigator.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
