"""
maths.py

Small numeric helpers used across the project (describe, histogram, models).

Design goals:
- No dependency on NumPy.
- Deterministic results (pure functions).
- Missing values are handled *before* calling these helpers (these functions
  expect real floats).

Conventions:
- For statistics that are undefined with fewer than 2 values (sample std),
  functions return math.nan.
- In this project, mean/quartiles/min/max also return math.nan when the list
  has fewer than 2 values to match the subject’s expected behavior.
"""

from __future__ import annotations

import math
from typing import Sequence


class Maths:
    """Namespace (static-only) for numeric computations used by the scripts."""

    @staticmethod
    def mean(values: Sequence[float]) -> float:
        """
        Compute the arithmetic mean.
        Args:
            values: Sequence of floats.
        Returns:
            Mean of `values`, or math.nan if there are fewer than 2 values.
        """
        total_entries = len(values)
        if total_entries < 2:
            return math.nan
        return sum(values) / total_entries

    @staticmethod
    def variance(values: Sequence[float]) -> float:
        """
        Return sample variance of values.
        Returns:
            math.nan if values empty
            0.0 if length == 1
        """
        n = len(values)
        if n == 0:
            return math.nan
        if n == 1:
            return 0.0

        mean_value = Maths.mean(values)

        total = 0.0
        for v in values:
            diff = v - mean_value
            total += diff * diff

        return total / (n - 1)

    @staticmethod
    def std(values: Sequence[float]) -> float:
        """
        Compute the sample standard deviation (denominator n-1).
        Args:
            values: Sequence of floats.
        Returns:
            Sample standard deviation, or math.nan if fewer than 2 values.
        """
        variance = Maths.variance(values)
        return math.sqrt(variance)

    @staticmethod
    def quartile(values: Sequence[float], quartile: float) -> float:
        """
        Compute a quartile using linear interpolation.
        Args:
            values: Sequence of floats.
            quartile: Quantile in [0.0, 1.0] (e.g. 0.25, 0.50, 0.75).
        Returns:
            The requested quartile, or math.nan if fewer than 2 values.
        """
        total_entries = len(values)
        if total_entries < 2:
            return math.nan

        sorted_values = sorted(values)
        position = (total_entries - 1) * quartile

        high_index = int(math.ceil(position))
        low_index = int(math.floor(position))

        if high_index == low_index:
            return sorted_values[high_index]

        weight = position - low_index
        return sorted_values[low_index] + weight * (
            sorted_values[high_index] - sorted_values[low_index]
            )

    @staticmethod
    def min_max(values: Sequence[float]) -> tuple[float, float]:
        """
        Compute minimum and maximum values.
        Args:
            values: Sequence of floats.
        Returns:
            (min, max) as floats, or math.nan if fewer than 2 values.
        """
        if len(values) < 2:
            return (math.nan, math.nan)

        minimum = values[0]
        maximum = values[0]

        for value in values:
            if value < minimum:
                minimum = value
            elif value > maximum:
                maximum = value

        return (minimum, maximum)

    @staticmethod
    def group_means(grouped_values: dict[str,
                                         list[float]
                                         ]) -> dict[str, float]:
        """
        Compute mean for each group(Houses).
        Empty groups return math.nan.
        """
        out: dict[str, float] = {}

        for group, values in grouped_values.items():
            if values:
                out[group] = Maths.mean(values)
            else:
                out[group] = math.nan

        return out

    @staticmethod
    def group_stds(grouped_values: dict[str, list[float]]) -> dict[str, float]:
        """
        Compute sample standard deviation per group.
        Empty groups return math.nan.
        """
        out: dict[str, float] = {}

        for group, values in grouped_values.items():
            if values:
                out[group] = Maths.std(values)
            else:
                out[group] = math.nan

        return out

    @staticmethod
    def between_class_variance(grouped_values: dict[str,
                                                    list[float]
                                                    ]) -> float:
        """
        Weighted variance of class means around the global mean.
        Returns:
            math.nan if no data.
        Note : Tells how "far" a class mean is compare to global mean
        """
        all_values: list[float] = []
        for vals in grouped_values.values():
            all_values.extend(vals)

        if not all_values:
            return math.nan

        global_mean = Maths.mean(all_values)

        total = 0.0
        total_count = 0

        for vals in grouped_values.values():
            if not vals:
                continue

            mean_val = Maths.mean(vals)
            count = len(vals)

            diff = mean_val - global_mean
            total += count * diff * diff
            total_count += count

        if total_count == 0:
            return math.nan

        return total / total_count

    @staticmethod
    def within_class_variance(grouped_values: dict[str, list[float]]) -> float:
        """
        Weighted average of variances inside each group.
        Returns :
            math.nan is no data
        Note : 
            Tells how "spread" values are inside each class
        """
        total = 0.0
        total_weight = 0

        for vals in grouped_values.values():
            if not vals:
                continue

            var = Maths.variance(vals)
            count = len(vals)

            if math.isnan(var):
                continue

            total += count * var
            total_weight += count

        if total_weight == 0:
            return math.nan

        return total / total_weight

    @staticmethod
    def separation_score(grouped_values: dict[str, list[float]]) -> float:
        """
        Ratio of between-class variance to within-class variance.

        Higher values indicate stronger class separation.
        """
        between = Maths.between_class_variance(grouped_values)
        within = Maths.within_class_variance(grouped_values)

        if math.isnan(between) or math.isnan(within):
            return math.nan

        if within == 0.0:
            if between == 0.0:
                return 0.0
            return math.inf

        return between / within

    @staticmethod
    def mean_spread(group_means: dict[str, float]) -> float:
        """
        Tells how wide means are spread between houses for a given feature
        """
        if not group_means:
            return 0.0

        values = list(group_means.values())
        minimum, maximum = Maths.min_max(values)
        return maximum - minimum

    @staticmethod
    def avgr_std_group(group_stds: dict[str, float]) -> float:
        """
        Compute average derivation of all houses for a feature
        """
        if not group_stds:
            return 0.0
        values = list(group_stds.values())
        return Maths.mean(values)

    @staticmethod
    def norm_spread(
            group_means: dict[str, float],
            group_stds: dict[str, float]
                ) -> float:
        """
        Normalize mean spread so it's "unit free"
        """
        if not group_means or not group_stds:
            return 0.0
        mean_spread = Maths.mean_spread(group_means)
        avg_std = Maths.avgr_std_group(group_stds)

        if mean_spread == 0.0 or avg_std == 0.0:
            return 0.0
        return mean_spread / avg_std

    @staticmethod
    def covariance(x: list[float], y: list[float]) -> float:
        """
        Sample covariance of two aligned vectors.

        Returns math.nan if invalid.
        """
        n = len(x)

        if n != len(y) or n < 2:
            return math.nan

        mean_x = Maths.mean(x)
        mean_y = Maths.mean(y)

        total = 0.0

        for xi, yi in zip(x, y):
            total += (xi - mean_x) * (yi - mean_y)

        return total / (n - 1)

    @staticmethod
    def correlation(x: list[float], y: list[float]) -> float:
        """
        Pearson correlation coefficient in [-1, 1].
        """
        cov = Maths.covariance(x, y)

        std_x = Maths.std(x) if x else math.nan
        std_y = Maths.std(y) if y else math.nan

        if math.isnan(cov) or math.isnan(std_x) or math.isnan(std_y):
            return math.nan

        if std_x == 0.0 or std_y == 0.0:
            return math.nan

        return cov / (std_x * std_y)
