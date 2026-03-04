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
    def std(values: Sequence[float]) -> float:
        """
        Compute the sample standard deviation (denominator n-1).

        Args:
            values: Sequence of floats.

        Returns:
            Sample standard deviation, or math.nan if fewer than 2 values.
        """
        mean_value = Maths.mean(values)
        if math.isnan(mean_value):
            return math.nan

        total_delta_square = 0.0
        for value in values:
            delta = value - mean_value
            total_delta_square += delta * delta

        variance = total_delta_square / (len(values) - 1)
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

        Notes:
            Uses the common "position = (n - 1) * q" interpolation approach.
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
        return sorted_values[low_index] + weight * (sorted_values[high_index] - sorted_values[low_index])

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


# TO BE REMOVE ONCE FULLY MERGE ?
# Compatibility layer (lets old imports keep working)
def our_mean(values: Sequence[float]) -> float:
    return Maths.mean(values)


def our_std(values: Sequence[float]) -> float:
    return Maths.std(values)


def our_quartile(values: Sequence[float], quartile: float) -> float:
    return Maths.quartile(values, quartile)


def our_min_max(values: Sequence[float]) -> tuple[float, float]:
    return Maths.min_max(values)