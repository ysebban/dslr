"""
pair_plot.py

Pair-plot (scatter matrix) in "one page per feature" mode.

Each page:
- base feature (the current page)
- a grid of plots:
    - histogram for the base feature
    - scatter plots: base feature vs every other feature (colored by house)
    - each scatter subplot shows correlation for that pair

Usage:
  python pair_plot.py <csv_path>
"""

# from __future__ import annotations

import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

from CsvManip import CsvManip
from PlotNavigator import PlotNavigator


HOUSES = ("Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff")


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
        """
        Initialize a pair-plot helper.

        Args:
            dataframe: Source pandas DataFrame.
        """
        self.dataframe = dataframe

        features = CsvManip.loadFeatures(self.dataframe)
        self.feature_names = list(features.keys())

        self.columns_count = 4
        self.rows_count = max(
            1,
            math.ceil(len(self.feature_names) / self.columns_count)
            )

    def make_figure(self):
        """
        Create the matplotlib figure used for one pair-plot page.

        Returns:
            A matplotlib figure.
        """
        return plt.figure(figsize=(
            3.5 * self.columns_count, 3 * self.rows_count
            ))

    def make_axes(self, figure):
        """
        Create the subplot axes for one page.

        Args:
            figure: Matplotlib figure where axes must be created.

        Returns:
            A flat tuple of axes.
        """

        figure.subplots_adjust(wspace=0.4, hspace=0.5)
        axes = figure.subplots(self.rows_count, self.columns_count)
        return tuple(axes.flat) if hasattr(axes, "flat") else (axes,)

    def render(self, base_feature: str, axes, index: int, total: int) -> str:
        """
        Render one page of the pair plot for a base feature.

        Args:
            base_feature: Current base feature.
            axes: Flat tuple of matplotlib axes.
            index: Current page index.
            total: Total number of pages.

        Returns:
            The base feature name, used by PlotNavigator in the title.
        """
        self._hide_all_axes(axes)

        for plot_index, other_feature in enumerate(self.feature_names):
            plot_axis = axes[plot_index]
            plot_axis.set_visible(True)

            if other_feature == base_feature:
                self._draw_histogram_subplot(plot_axis, base_feature)
            else:
                self._draw_scatter_subplot(
                    plot_axis,
                    base_feature,
                    other_feature,
                )

        return base_feature

    def _hide_all_axes(self, axes) -> None:
        """
        Hide all subplot axes before drawing the current page.
        """
        for plot_axis in axes:
            plot_axis.set_visible(False)

    def _draw_histogram_subplot(self, plot_axis, feature_name: str) -> None:
        """
        Draw the histogram subplot for the base feature.
        """
        _, matrix, house_labels = CsvManip.loadFeaturesMatrix(
            self.dataframe,
            [feature_name],
            labels=True,
        )

        if not matrix:
            plot_axis.set_title(f"{feature_name} (hist)")
            plot_axis.set_xlabel(feature_name)
            plot_axis.set_ylabel("Count")
            return

        if house_labels is not None:
            for house_name in HOUSES:
                house_values = self._house_hist_values(
                    matrix,
                    house_labels,
                    house_name
                )
                plot_axis.hist(
                    house_values,
                    bins=20,
                    alpha=0.5,
                    density=False,
                    label=house_name,
                )
            plot_axis.legend(fontsize=8)
        else:
            values = [row[0] for row in matrix]
            plot_axis.hist(values, bins=20, alpha=0.7, density=False)

        plot_axis.set_title(f"{feature_name} (hist)")
        plot_axis.set_xlabel(feature_name)
        plot_axis.set_ylabel("Count")

    def _draw_scatter_subplot(
        self,
        plot_axis,
        base_feature: str,
        other_feature: str,
    ) -> None:
        """
        Draw one scatter subplot for a feature pair.
        """
        _, matrix, house_labels = CsvManip.loadFeaturesMatrix(
            self.dataframe,
            [other_feature, base_feature],
            labels=True,
        )

        if not matrix:
            plot_axis.set_title(f"{other_feature} (corr=nan)", fontsize=9)
            plot_axis.set_xlabel(other_feature, fontsize=8)
            plot_axis.set_ylabel(base_feature, fontsize=8)
            return

        x_values = [row[0] for row in matrix]
        y_values = [row[1] for row in matrix]

        if house_labels is not None:
            self._draw_points_by_house(
                plot_axis,
                matrix,
                house_labels,
            )
        else:
            plot_axis.scatter(x_values, y_values, s=10, alpha=0.65)

        correlation_text = self._correlation_text(x_values, y_values)

        plot_axis.set_title(f"{other_feature} (corr={correlation_text})",
                            fontsize=8
                            )
        plot_axis.set_xlabel(other_feature, fontsize=6)
        plot_axis.set_ylabel(base_feature, fontsize=6)

    def _draw_points_by_house(
        self,
        plot_axis,
        matrix: list[list[float]],
        house_labels: list[str],
    ) -> None:
        """
        Draw scatter points grouped by house.
        """
        for house_name in HOUSES:
            house_x, house_y = self._house_scatter_points(
                matrix,
                house_labels,
                house_name,
            )
            plot_axis.scatter(house_x, house_y, s=5, alpha=0.65)

    def _house_hist_values(
        self,
        matrix: list[list[float]],
        house_labels: list[str],
        house_name: str,
    ) -> list[float]:
        """
        Extract histogram values for one house.
        """
        values: list[float] = []

        for row_values, label in zip(matrix, house_labels):
            if label == house_name:
                values.append(row_values[0])

        return values

    def _house_scatter_points(
        self,
        matrix: list[list[float]],
        house_labels: list[str],
        house_name: str,
    ) -> tuple[list[float], list[float]]:
        """
        Extract scatter X/Y points for one house.
        """
        house_x: list[float] = []
        house_y: list[float] = []

        for row_values, label in zip(matrix, house_labels):
            if label == house_name:
                house_x.append(row_values[0])
                house_y.append(row_values[1])

        return house_x, house_y

    def _correlation_text(
        self,
        x_values: list[float],
        y_values: list[float],
    ) -> str:
        """
        Compute and format correlation text for one feature pair.
        """
        correlation = pd.Series(x_values).corr(pd.Series(y_values))
        return "nan" if correlation != correlation else f"{correlation:.3f}"


def main(argv: list[str] | None = None) -> int:
    """
    Run the pair-plot viewer from command line arguments.

    Args:
        argv: Optional argument list. If None, uses command line arguments.

    Returns:
        Exit status code.
    """
    parser = argparse.ArgumentParser(
        prog="pair_plot.py",
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
        plot.feature_names,
        render=plot.render,
        make_figure=plot.make_figure,
        make_axes=plot.make_axes,
        title="Pair plot",
    )
    navigator.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
