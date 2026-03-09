"""
describe_bonus.py

Extended statistical analysis for numeric features in the Hogwarts dataset.

This module builds on top of the base `describe.py` report and introduces
additional metrics useful for feature analysis in classification problems.

Additional metrics include:
- Mean spread between houses
- Normalized spread (spread relative to class variance)
- Missing value statistics
- Feature separation score

It also provides higher-level dataset diagnostics such as:
- Feature ranking by predictive strength
- Feature-level insights
- Global dataset diagnostics

Usage:
  python describe_bonus.py <csv_path> [-b] [-s]
"""

from mandatory.describe import FeatureMetrics, DescribeReport
from utils.maths import Maths
from utils.CsvManip import CsvManip

import argparse
from dataclasses import dataclass


BONUS_LABELS: dict[str, str] = {
    "mean_spread": "Mean spread",
    "norm_spread": "Norm spread",
    "missing_count": "Missing",
    "missing_ratio": "Missing %",
    "separation_score": "Separation",
}


# ---------- Bonus Metrics ----------
@dataclass(frozen=True, slots=True)
class BonusFeatureMetrics(FeatureMetrics):
    """
    Extended metrics for a single feature.

    Inherits base descriptive statistics from `FeatureMetrics`
    and adds additional metrics useful for classification analysis.

    Attributes:
        mean_spread:
            Difference between the highest and lowest class mean.

        norm_spread:
            Normalized spread relative to average class standard deviation.

        missing_count:
            Number of missing values for this feature.

        missing_ratio:
            Percentage of missing values in the dataset.

        group_stds:
            Per-house standard deviations.

        group_means:
            Per-house means.

        separation_score:
            Measure of how well the feature separates houses.
    """

    mean_spread: float
    norm_spread: float
    missing_count: int
    missing_ratio: float
    group_stds: dict[str, float]
    group_means: dict[str, float]
    separation_score: float

    @classmethod
    def from_values(
        cls,
        feature_values: list[float],
        missing_values: tuple[int, int],
        groups: dict[str, list[float]]
    ) -> "BonusFeatureMetrics":
        """
        Build extended metrics for a feature.

        Args:
            feature_values:
                Clean numeric values for the feature.

            missing_values:
                Tuple (missing_count, total_count).

            groups:
                Mapping {house -> list of numeric values}.

        Returns:
            BonusFeatureMetrics instance containing base statistics
            and additional class-aware metrics.
        """

        missing_count, full_length = missing_values

        # Base descriptive statistics
        base = FeatureMetrics.from_values(feature_values)

        missing_ratio = (missing_count / full_length) * 100

        # Class-aware statistics
        means_groups = Maths.group_means(groups)
        stds_groups = Maths.group_stds(groups)

        return cls(
            count=base.count,
            mean=base.mean,
            std=base.std,
            min=base.min,
            q1=base.q1,
            q2=base.q2,
            q3=base.q3,
            max=base.max,

            mean_spread=Maths.mean_spread(means_groups),
            norm_spread=Maths.norm_spread(means_groups, stds_groups),

            missing_count=missing_count,
            missing_ratio=missing_ratio,

            group_stds=stds_groups,
            group_means=means_groups,

            separation_score=Maths.separation_score(groups),
        )


# ---------- Bonus Report ----------
@dataclass(slots=True)
class BonusReport(DescribeReport):
    """
    Extended report built on top of `DescribeReport`.

    This report contains:
    - Base statistics inherited from `DescribeReport`
    - Bonus classification-oriented metrics
    """

    by_bonus: dict[str, BonusFeatureMetrics]

    @classmethod
    def from_features(
        cls,
        features: dict[str, list[float]],
        missing: dict[str, float],
        groups: dict[str, dict[str, list[float]]]
    ) -> "BonusReport":
        """
        Build the bonus report from extracted features.

        Args:
            features:
                Mapping {feature -> numeric values}

            missing:
                Mapping {feature -> (missing_count, total_count)}

            groups:
                Mapping {feature -> {house -> values}}

        Returns:
            Fully constructed BonusReport.
        """

        by_feature: dict[str, FeatureMetrics] = {}
        by_bonus: dict[str, BonusFeatureMetrics] = {}

        for feature_name, values in features.items():

            base_metrics = FeatureMetrics.from_values(values)

            bonus_metrics = BonusFeatureMetrics.from_values(
                features[feature_name],
                missing[feature_name],
                groups[feature_name],
            )

            by_feature[feature_name] = base_metrics
            by_bonus[feature_name] = bonus_metrics

        return cls(
            by_feature=by_feature,
            by_bonus=by_bonus,
        )

    # ---------- Feature ranking ----------
    def print_feature_ranking(self) -> None:
        """
        Print a ranking of features based on their separation score.
        """

        print("\tFeature Ranking\n")

        rows = []

        for feature, metrics in self.by_bonus.items():
            rows.append(
                (
                    feature,
                    metrics.separation_score,
                    metrics.missing_ratio,
                )
            )

        rows.sort(key=lambda r: r[1], reverse=True)

        header = f"{'Feature':<26} {'SepScore':>8} {'Missing%':>9}  Quality"
        print(header)
        print("-" * len(header))

        for feature, sep, miss in rows:

            bar_len = min(int(sep), 10)
            bar = "█" * bar_len

            if sep >= 6:
                label = "STRONG"
            elif sep >= 4:
                label = "GOOD"
            elif sep >= 2:
                label = "OK"
            elif sep >= 1:
                label = "WEAK"
            else:
                label = "POOR"

            print(
                f"{feature:<26} "
                f"{sep:8.2f} "
                f"{miss:9.2f}  "
                f"{bar:<10} {label}"
            )

    # ---------- Feature insights ----------
    def print_feature_insights(self) -> None:
        """
        Print per-feature diagnostics explaining how houses differ.
        """

        print("\n\tFeature Insights\n")

        for feature, metrics in self.by_bonus.items():

            sep = metrics.separation_score

            if sep >= 6:
                strength = "excellent"
            elif sep >= 4:
                strength = "good"
            elif sep >= 2:
                strength = "moderate"
            else:
                strength = "weak"

            print(f"{feature}")
            print(f"  SepScore : {sep:.2f} ({strength})")

            print("  Houses   :")

            for house in metrics.group_means:

                mean = metrics.group_means[house]
                std = metrics.group_stds[house]

                print(f"      {house:<12} {mean:8.2f} ±{std:6.2f}")

            if sep < 1:
                print("\tInsight  : values almost identical between houses.")
            elif sep < 3:
                print("\tInsight  : partial separation but strong overlaps.")
            else:
                print("\tInsight  : clear clustering between houses.")

            print()

    # ---------- Dataset diagnostics ----------
    def print_dataset_diagnostics(self) -> None:
        """
        Print a global overview of feature predictive strength.
        """

        print("\n\tDataset Diagnostics\n")

        strong = good = ok = weak = poor = 0
        missing_values = []

        for metrics in self.by_bonus.values():

            sep = metrics.separation_score
            missing_values.append(metrics.missing_ratio)

            if sep >= 6:
                strong += 1
            elif sep >= 4:
                good += 1
            elif sep >= 2:
                ok += 1
            elif sep >= 1:
                weak += 1
            else:
                poor += 1

        total = len(self.by_bonus)

        print(f"Total features analyzed : {total}")
        print(f"Strong predictors       : {strong}")
        print(f"Good predictors         : {good}")
        print(f"Moderate predictors     : {ok}")
        print(f"Weak predictors         : {weak}")
        print(f"Poor predictors         : {poor}")

        if missing_values:
            print(
                f"Missing values range    : "
                f"{min(missing_values):.2f}% → {max(missing_values):.2f}%"
            )

    # ---------- Combined summary ----------
    def print_summary(self) -> None:
        """
        Print the full bonus analysis summary.
        """

        self.print_feature_ranking()
        self.print_feature_insights()
        self.print_dataset_diagnostics()


# ---------- CLI ----------
def parse_args(argv: list[str] | None = None):
    """
    Parse CLI arguments for the bonus report.
    """

    parser = argparse.ArgumentParser(
        prog="describe_bonus.py",
        description="Extended describe report with feature-selection hints.",
    )

    parser.add_argument("csv_path", help="Path to the CSV dataset.")
    parser.add_argument("-H", "--houses", nargs="+", default=None)
    parser.add_argument("-f", "--features", nargs="+", default=None)
    parser.add_argument("-s", "--summary", action="store_true")
    parser.add_argument("-b", "--base", action="store_true")

    args = parser.parse_args(argv)

    return (
        args.csv_path,
        args.houses,
        args.features,
        args.summary,
        args.base,
    )


# ---------- Main ----------
def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the extended describe CLI.

    Loads dataset, computes base and bonus metrics,
    and prints the requested reports.
    """

    path, houses, feature_names, summary, base = parse_args(argv)

    try:
        dataframe = CsvManip.loadCsv(path)
    except Exception as exc:
        print(f"Cannot load CSV: {exc!s}")
        return 1

    feature_values, missing_values = CsvManip.loadFeatures(
        dataframe,
        houses=houses if houses else None,
        select_cols=feature_names if feature_names else None
    )

    groups = CsvManip.build_groups(dataframe, houses, feature_names)

    report = BonusReport.from_features(
        feature_values,
        missing_values,
        groups
    )

    if base:
        print("\n==== Base Report ====\n")
        print(report)

    print("\n==== Bonus Report ====\n")
    print(report._render_metrics_table(report.by_bonus, BONUS_LABELS))

    if summary:
        print("\n==== Summary ====\n")
        report.print_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())