"""
describe.py

Compute and print summary statistics for numeric features in a CSV dataset.

The output is similar to `pandas.DataFrame.describe()` and includes:
Count, Mean, Std, Min, 25%, 50%, 75%, Max.

A column is considered a numeric feature if:
- Missing values are ignored (None, NaN, empty/whitespace strings).
- Every non-missing value can be converted to float.

Usage:
  python describe.py <csv_path>
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from typing import Iterable

from CsvManip import CsvManip
from maths import Maths


LABELS: dict[str, str] = {
    "count": "Count",
    "mean": "Mean",
    "std": "Std",
    "min": "Min",
    "q1": "25%",
    "q2": "50%",
    "q3": "75%",
    "max": "Max",
}

PRECISION = 6          # numeric formatting precision
MAX_HEADER_W = 18      # maximum visible characters for feature headers
MIN_COLUMN_W = 12      # minimum width of a feature column in characters


# ---------- Metrics ----------
@dataclass(frozen=True, slots=True)
class FeatureMetrics:
    """
    Summary statistics for a single numeric feature.

    Attributes:
        count: Number of numeric values (as float for printing consistency).
        mean: Arithmetic mean.
        std: Sample standard deviation (n-1).
        min: Minimum value.
        q1:  25th percentile.
        q2:  50th percentile (median).
        q3:  75th percentile.
        max: Maximum value.
    """

    count: float
    mean: float
    std: float
    min: float
    q1: float
    q2: float
    q3: float
    max: float

    @classmethod
    def from_values(cls, values: list[float]) -> "FeatureMetrics":
        """
        Compute metrics from a list of floats.

        Args:
            values: List of numeric values (already converted to float).

        Returns:
            FeatureMetrics instance. If `values` is empty, all metrics are NaN.
        """
        if not values:
            nan = math.nan
            return cls(nan, nan, nan, nan, nan, nan, nan, nan)

        minimum, maximum = Maths.min_max(values)
        return cls(
            count=float(len(values)),
            mean=Maths.mean(values),
            std=Maths.std(values),
            min=minimum,
            q1=Maths.quartile(values, 0.25),
            q2=Maths.quartile(values, 0.50),
            q3=Maths.quartile(values, 0.75),
            max=maximum,
        )


# ---------- Report rendering ----------
@dataclass(frozen=True, slots=True)
class FeatureColumn:
    """Rendering metadata for one feature column."""

    feature_name: str
    header: str
    width: int


@dataclass(slots=True)
class DescribeReport:
    """
    Terminal-friendly table of FeatureMetrics for multiple features.

    This class only formats computed metrics. Feature extraction is performed
    elsewhere and passed in as a mapping {feature_name -> list[float]}.
    """

    by_feature: dict[str, FeatureMetrics]

    # ---------- Construction ----------
    @classmethod
    def from_features(cls,
                      features: dict[str, list[float]]
                      ) -> "DescribeReport":
        """
        Build a report from pre-extracted numeric features.

        Args:
            features: Mapping {feature_name -> list of float values}.

        Returns:
            DescribeReport containing metrics for every feature.
        """
        metrics_by_feature: dict[str, FeatureMetrics] = {}
        for feature_name, values in features.items():
            metrics_by_feature[feature_name] = \
                FeatureMetrics.from_values(values)
        return cls(metrics_by_feature)

    # ---------- Layout / formatting helpers ----------
    @staticmethod
    def _terminal_width(default: int = 120) -> int:
        """Return terminal width with a safe fallback."""
        return shutil.get_terminal_size((default, 20)).columns

    @staticmethod
    def _metric_label_width(metric_names: Iterable[str]) -> int:
        """Width of the left label column."""
        return max(
            len(
                LABELS.get(metric_name, metric_name)
                ) for metric_name in metric_names
            )

    @staticmethod
    def _truncate_feature_name(feature_name: str) -> str:
        """Truncate long feature names to keep headers readable."""
        if len(feature_name) <= MAX_HEADER_W:
            return feature_name
        return feature_name[: MAX_HEADER_W - 3] + "..."

    def _build_feature_columns(self) -> list[FeatureColumn]:
        """Build per-feature rendering metadata."""
        columns: list[FeatureColumn] = []
        for feature_name in self.by_feature.keys():
            header = self._truncate_feature_name(feature_name)
            width = max(MIN_COLUMN_W, len(header))
            columns.append(FeatureColumn(feature_name=feature_name,
                                         header=header,
                                         width=width
                                         ))
        return columns

    @staticmethod
    def _split_columns_to_fit_terminal(
        columns: list[FeatureColumn],
        *,
        terminal_width: int,
        metric_label_width: int,
    ) -> list[list[FeatureColumn]]:
        """Group columns into blocks that fit within the terminal width."""
        blocks: list[list[FeatureColumn]] = []

        current_block: list[FeatureColumn] = []
        used_width = metric_label_width + 1

        for column in columns:
            needed = column.width + 1
            if current_block and (used_width + needed > terminal_width):
                blocks.append(current_block)
                current_block = []
                used_width = metric_label_width + 1

            current_block.append(column)
            used_width += needed

        if current_block:
            blocks.append(current_block)

        return blocks

    @staticmethod
    def _format_number(value: float, width: int) -> str:
        """Format a float into a fixed-width cell."""
        if isinstance(value, float) and math.isnan(value):
            return f"{'nan':>{width}}"
        return f"{value:>{width}.{PRECISION}f}"

    @staticmethod
    def _render_header_row(
        block: list[FeatureColumn],
        metric_label_width: int
         ) -> str:
        """Render the header row for a column block."""
        left_padding = " " * (metric_label_width + 1)
        headers = " ".join(
            f"{column.header:>{column.width}}" for column in block
            )
        return left_padding + headers

    def _render_metric_row(
        self,
        metric_name: str,
        *,
        block: list[FeatureColumn],
        metric_label_width: int,
    ) -> str:
        """Render one metric row for a column block."""
        label = LABELS.get(metric_name, metric_name)
        parts = [f"{label:<{metric_label_width}}"]

        for column in block:
            metrics = self.by_feature[column.feature_name]
            value = getattr(metrics, metric_name)
            parts.append(self._format_number(value, column.width))

        return " ".join(parts)

    def _render_blocks(
        self,
        blocks: list[list[FeatureColumn]],
        *,
        metric_names: list[str],
        metric_label_width: int,
    ) -> str:
        """Render all blocks into the final report string."""
        lines: list[str] = []

        for block_index, block in enumerate(blocks):
            if block_index > 0:
                lines.append("")

            lines.append(self._render_header_row(block, metric_label_width))
            for metric_name in metric_names:
                lines.append(
                    self._render_metric_row(
                        metric_name,
                        block=block,
                        metric_label_width=metric_label_width,
                    )
                )

        return "\n".join(lines)

    # ---------- Public rendering ----------
    def __str__(self) -> str:
        """
        Do

        Args:
            
        Returns:
        
        """
        if not self.by_feature:
            return "No numeric features found."

        metric_names = LABELS.keys()
        metric_label_width = self._metric_label_width(metric_names)

        terminal_width = self._terminal_width()
        columns = self._build_feature_columns()
        blocks = self._split_columns_to_fit_terminal(
            columns,
            terminal_width=terminal_width,
            metric_label_width=metric_label_width,
        )

        return self._render_blocks(
            blocks,
            metric_names=metric_names,
            metric_label_width=metric_label_width,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="describe.py",
        description="Summary statistics for numeric features in a CSV.",
    )
    parser.add_argument("csv_path", help="Path to the CSV dataset.")
    args = parser.parse_args(argv)

    try:
        dataframe = CsvManip.loadCsv(args.csv_path)
    except Exception as exc:
        parser.error(f"Cannot load CSV: {exc!s}")
        return 1

    features = CsvManip.loadFeatures(dataframe)
    report = DescribeReport.from_features(features)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
