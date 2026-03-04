"""
scatter_plot.py

Grid scatter browser to find similar features.

Each page:
- One base feature (X axis)
- Grid of scatter plots: base feature vs every other feature (Y axis)
- Points colored by house
- Each subplot shows correlation for that pair

Navigation:
- Right arrow: next base feature
- Left arrow: previous base feature
- r: redraw
- q / escape: quit

Usage:
  python scatter_plot.py <csv_path>
"""

from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

from CsvManip import CsvManip
from PlotNavigator import PlotNavigator


HOUSES = ("Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff")
HOUSE_COL = "Hogwarts House"

IGNORE_COLS = {
    "index",
    "Index",
    "Hogwarts House",
    "First Name",
    "Last Name",
    "Birthday",
    "Best Hand",
}


class ScatterGridPlot:
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe

        features = CsvManip.loadFeatures(self.dataframe)
        self.feature_names = list(features.keys())

        # keep aligned numeric columns in a DataFrame
        self.numeric = self.dataframe[self.feature_names].apply(
                        pd.to_numeric, errors="coerce")

        self.houses = None
        if HOUSE_COL in self.dataframe.columns:
            self.houses = self.dataframe[HOUSE_COL]

    def make_figure(self):
        # One page = base feature vs (n-1) other features
        plots_count = max(0, len(self.feature_names) - 1)
        cols = 4
        rows = max(1, math.ceil(plots_count / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
        # Flatten axes to a tuple so PlotNavigator can iterate/clear them
        axes_flat = tuple(axes.flat) if hasattr(axes, "flat") else (axes,)
        return fig, axes_flat

    def render(self, base_feature: str, axes, index: int, total: int) -> str:
        # Hide all axes first
        for ax in axes:
            ax.set_visible(False)

        x_all = self.numeric[base_feature]

        other_features = [f for f in self.feature_names if f != base_feature]

        for plot_index, other_feature in enumerate(other_features):
            ax = axes[plot_index]
            ax.set_visible(True)

            y_all = self.numeric[other_feature]

            if self.houses is not None:
                for house in HOUSES:
                    mask_house = self.houses == house
                    valid = mask_house & x_all.notna() & y_all.notna()
                    ax.scatter(
                        x_all[valid],
                        y_all[valid],
                        s=10,
                        alpha=0.65,
                        label=house if plot_index == 0 else None,
                    )
                if plot_index == 0:
                    ax.legend()
            else:
                valid = x_all.notna() & y_all.notna()
                ax.scatter(x_all[valid], y_all[valid], s=10, alpha=0.65)

            corr = float(x_all.corr(y_all))
            corr_text = "nan" if corr != corr else f"{corr:.3f}"

            ax.set_title(f"{other_feature}  (corr={corr_text})")
            ax.set_xlabel(base_feature)
            ax.set_ylabel(other_feature)

        # This string becomes the PlotNavigator header
        return base_feature


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scatter_plot.py",
        description="Grid scatter browser to find similar features.",
    )
    parser.add_argument("csv_path", help="Path to the CSV dataset.")
    args = parser.parse_args(argv)

    try:
        dataframe = CsvManip.loadCsv(args.csv_path)
    except Exception as exc:
        parser.error(f"Cannot load CSV: {exc!s}")
        return 1

    plot = ScatterGridPlot(dataframe)
    if len(plot.feature_names) < 2:
        print("Not enough numeric features to build scatter plots.")
        return 0

    navigator = PlotNavigator(
        plot.feature_names,          # pages = base features
        render=plot.render,          # bound method
        make_figure=plot.make_figure,
        title="Scatter plot",
    )
    navigator.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())