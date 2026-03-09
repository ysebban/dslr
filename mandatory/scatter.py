"""
scatter.py

Grid scatter browser to find similar features.

Each page:
- One base feature (X axis)
- Grid of scatter plots: base feature vs every other feature (Y axis)
- Points colored by house
- Each subplot shows correlation for that pair

Usage:
  python scatter.py <csv_path>
"""
import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd

from utils.CsvManip import CsvManip
from utils.PlotNavigator import PlotNavigator


HOUSES = ("Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff")


class ScatterGridPlot:
    """
    Scatter plot utilities for this project.

    This class displays one base feature per page.
    Each subplot compares the base feature with one other feature.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialize a scatter plot helper.

        Args:
            dataframe: Source pandas DataFrame.

        Notes:
            Only numeric feature names are stored here.
            Pair alignment is delegated to CsvManip.loadFeaturesMatrix().
        """
        self.dataframe = dataframe

        features = CsvManip.loadFeatures(self.dataframe)
        self.feature_names = list(features.keys())

    def make_figure(self):
        return plt.figure(figsize=(12, 8))

    def make_axes(self, figure):
        plots_count = max(0, len(self.feature_names) - 1)
        columns_count = 4
        rows_count = max(1, math.ceil(plots_count / columns_count))

        figure.subplots_adjust(wspace=0.4, hspace=0.5)
        axes = figure.subplots(rows_count, columns_count)

        return tuple(axes.flat) if hasattr(axes, "flat") else (axes,)

    def render(self, base_feature: str, axes, index: int, total: int) -> str:
        """
        Render one page of scatter plots for a base feature.
        """
        self._hide_all_axes(axes)

        for plot_index, other_feature in enumerate(
            self._other_features(base_feature)
        ):
            plot_axis = axes[plot_index]
            plot_axis.set_visible(True)

            _, matrix, house_labels = self._load_pair_data(
                base_feature,
                other_feature,
            )

            self._draw_pair_subplot(
                plot_axis,
                base_feature,
                other_feature,
                matrix,
                house_labels,
                show_legend=(plot_index == 0),
            )

        return base_feature

    def _hide_all_axes(self, axes) -> None:
        """
        Hide all subplot axes before drawing the current page.
        """
        for plot_axis in axes:
            plot_axis.set_visible(False)

    def _other_features(self, base_feature: str) -> list[str]:
        """
        Return all features except the current base feature.
        """
        return [
            feature_name
            for feature_name in self.feature_names
            if feature_name != base_feature
        ]

    def _load_pair_data(
        self,
        base_feature: str,
        other_feature: str,
    ) -> tuple[list[str], list[list[float]], list[str] | None]:
        """
        Load row-aligned data for one feature pair.
        """
        return CsvManip.loadFeaturesMatrix(
            self.dataframe,
            [base_feature, other_feature],
            labels=True,
        )

    def _draw_pair_subplot(
        self,
        plot_axis,
        base_feature: str,
        other_feature: str,
        matrix: list[list[float]],
        house_labels: list[str] | None,
        *,
        show_legend: bool,
    ) -> None:
        """
        Draw one subplot for one feature pair.
        """
        if not matrix:
            self._style_empty_axis(plot_axis, base_feature, other_feature)
            return

        base_values = [row[0] for row in matrix]
        other_values = [row[1] for row in matrix]

        if house_labels is not None:
            self._draw_points_by_house(
                plot_axis,
                matrix,
                house_labels,
                show_legend=show_legend,
            )
        else:
            plot_axis.plot(base_values, other_values, ".", markersize=2)

        correlation_text = self._correlation_text(base_values, other_values)

        plot_axis.set_title(f"{other_feature}  (corr={correlation_text})")
        plot_axis.set_xlabel(base_feature)
        plot_axis.set_ylabel(other_feature)

    def _style_empty_axis(
        self,
        plot_axis,
        base_feature: str,
        other_feature: str,
    ) -> None:
        """
        Style one subplot when no valid pair data is available.
        """
        plot_axis.set_title(f"{other_feature}  (corr=nan)")
        plot_axis.set_xlabel(base_feature)
        plot_axis.set_ylabel(other_feature)

    def _draw_points_by_house(
        self,
        plot_axis,
        matrix: list[list[float]],
        house_labels: list[str],
        *,
        show_legend: bool,
    ) -> None:
        """
        Draw one subplot with points grouped by house.
        """
        for house_name in HOUSES:
            house_x, house_y = self._house_points(
                matrix,
                house_labels,
                house_name,
            )

            plot_axis.scatter(
                house_x,
                house_y,
                s=6,
                alpha=0.65,
                label=house_name if show_legend else None,
            )

        if show_legend:
            plot_axis.legend()

    def _house_points(
        self,
        matrix: list[list[float]],
        house_labels: list[str],
        house_name: str,
    ) -> tuple[list[float], list[float]]:
        """
        Extract X/Y points belonging to one house.
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
        base_values: list[float],
        other_values: list[float],
    ) -> str:
        """
        Compute and format correlation text for one feature pair.
        """
        correlation = pd.Series(base_values).corr(pd.Series(other_values))
        return "nan" if correlation != correlation else f"{correlation:.3f}"


def main(argv: list[str] | None = None) -> int:
    """
    Run the scatter plot viewer from command line arguments.

    Args:
        argv: Optional argument list. If None, uses command line arguments.

    Returns:
        Exit status code.
    """
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
        plot.feature_names,
        render=plot.render,
        make_figure=plot.make_figure,
        make_axes=plot.make_axes,
        title="Scatter plot",
    )
    navigator.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
