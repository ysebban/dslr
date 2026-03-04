"""
histogram.py

Interactive viewer for per-house distributions (Mean + Std per feature).

Navigation:
- Right arrow: next feature
- Left arrow: previous feature
- r: redraw
- q / escape: quit

Usage:
  python histogram.py <csv_path>
"""

from __future__ import annotations

import argparse
import math
import matplotlib.pyplot as plt

from CsvManip import CsvManip
from maths import Maths
from PlotNavigator import PlotNavigator


HOUSES = ("Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff")


class HistogramPlot:
    """
    Histogram viewer: for each feature, display mean and std by house.
    """

    def __init__(self, dataframe) -> None:
        self.dataframe = dataframe

        all_features = CsvManip.loadFeatures(self.dataframe)
        self.feature_names = list(all_features.keys())

        self.by_house = {
            house: CsvManip.loadFeatures(self.dataframe, house=house)
            for house in HOUSES
        }

    def make_figure(self):
        fig, (ax_mean, ax_std) = plt.subplots(1, 2, figsize=(12, 5))
        return fig, (ax_mean, ax_std)

    def render(self, feature_name: str, axes, index: int, total: int) -> str:
        ax_mean, ax_std = axes

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

        ax_mean.bar(HOUSES, means, color=colors)
        ax_mean.set_title("Mean by house")
        ax_mean.set_xlabel("House")
        ax_mean.set_ylabel("Mean")
        ax_mean.tick_params(axis="x", rotation=45)

        ax_std.bar(HOUSES, stds, color=colors)
        ax_std.set_title("Std by house")
        ax_std.set_xlabel("House")
        ax_std.set_ylabel("Std")
        ax_std.tick_params(axis="x", rotation=45)

        return feature_name


def main(argv: list[str] | None = None) -> int:
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
        render=histogram.render,            # bound method
        make_figure=histogram.make_figure,  # bound method
        title="Histogram",
    )
    navigator.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
