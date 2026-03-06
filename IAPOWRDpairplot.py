"""
pair_plot.py

Pair-plot (scatter matrix) in "one page per feature" mode.

Each page:
- base feature (the current page)
- a grid of plots:
    - histogram for the base feature
    - scatter plots: base feature vs every other feature (colored by house)
    - each scatter subplot shows correlation for that pair

Navigation (PlotNavigator):
- Right arrow: next base feature
- Left arrow: previous base feature
- r: redraw
- q / escape: quit

Usage:
  python pair_plot.py <csv_path>
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


class PairPlotByFeature:
    """
    One-page-per-feature pair plot.

    Items navigated by PlotNavigator:
      feature_names (list[str])

    For each base feature page:
      - histogram(base)
      - scatter(base vs other) for all other numeric features
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.dataframe = dataframe
        features = CsvManip.loadFeatures(self.dataframe)
        self.feature_names = list(features.keys())

        # Keep aligned numeric columns
        self.numeric = self.dataframe[self.feature_names].apply(
                                pd.to_numeric, errors="coerce")

        self.houses = self.dataframe[HOUSE_COL]\
            if HOUSE_COL in self.dataframe.columns else None

        # Grid layout (one cell per feature: 1 hist + (n-1) scatters)
        self.cols = 4
        self.rows = max(1, math.ceil(len(self.feature_names) / self.cols))

    def make_figure(self):
        fig, axes = plt.subplots(self.rows, self.cols, figsize=(4.0 * self.cols, 3.2 * self.rows))
        axes_flat = tuple(axes.flat) if hasattr(axes, "flat") else (axes,)
        return fig, axes_flat

    def render(self, base_feature: str, axes, index: int, total: int) -> str:
        # Hide all axes by default (useful if grid has extra empty slots).
        for ax in axes:
            ax.set_visible(False)

        base_values = self.numeric[base_feature]

        for plot_index, other_feature in enumerate(self.feature_names):
            ax = axes[plot_index]
            ax.set_visible(True)

            other_values = self.numeric[other_feature]

            # Diagonal cell (base == other): histogram
            if other_feature == base_feature:
                if self.houses is not None:
                    for house in HOUSES:
                        mask = self.houses == house
                        vals = base_values[mask & base_values.notna()]
                        ax.hist(vals, bins=20, alpha=0.5, density=False, label=house)
                    ax.legend(fontsize=8)
                else:
                    ax.hist(base_values[base_values.notna()], bins=20, alpha=0.7)

                ax.set_title(f"{base_feature} (hist)")
                ax.set_xlabel(base_feature)
                ax.set_ylabel("Count")
                continue

            # Off-diagonal: scatter (x = other, y = base)
            if self.houses is not None:
                for house in HOUSES:
                    mask = self.houses == house
                    valid = mask & base_values.notna() & other_values.notna()
                    ax.scatter(other_values[valid], base_values[valid], s=10, alpha=0.65)
            else:
                valid = base_values.notna() & other_values.notna()
                ax.scatter(other_values[valid], base_values[valid], s=10, alpha=0.65)

            corr = float(base_values.corr(other_values))
            corr_text = "nan" if corr != corr else f"{corr:.3f}"

            ax.set_title(f"{other_feature} (corr={corr_text})", fontsize=9)
            ax.set_xlabel(other_feature, fontsize=8)
            ax.set_ylabel(base_feature, fontsize=8)

        return base_feature


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pairplot.py",
        description="Pair plot in one-page-per-feature mode.",
    )
    parser.add_argument("csv_path", help="Path to the CSV dataset.")
    args = parser.parse_args(argv)

    try:
        dataframe = CsvManip.loadCsv(args.csv_path)
    except Exception as exc:
        parser.error(f"Cannot load CSV: {exc!s}")
        return 1

    plot = PairPlotByFeature(dataframe)
    if len(plot.feature_names) < 2:
        print("Not enough numeric features to build a pair plot.")
        return 0

    navigator = PlotNavigator(
        plot.feature_names,       # pages = base features
        render=plot.render,        # bound method
        make_figure=plot.make_figure,
        title="Pair plot",
    )
    navigator.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())