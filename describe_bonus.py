from describe import FeatureMetrics, FeatureColumn, DescribeReport
from maths import Maths
from CsvManip import CsvManip

import argparse
import pandas
from dataclasses import dataclass

BONUS_LABELS: dict[str, str] = {
    "variance": "Variance",
    "missing_count": "Missing",
    "missing_ratio": "Missing %",
    "separation_score": "Separation",
}


@dataclass(frozen=True, slots=True)
class BonusFeatureMetrics(FeatureMetrics):

    variance: float
    missing_count: int
    missing_ratio: float
    group_stds: dict[str, float]
    group_means: dict[str, float]
    separation_score: float

    @classmethod
    def from_dataframe(cls, dataframe: pandas.DataFrame, feature_name: str):

        raw_column = dataframe[feature_name].tolist()
        clean_column, missing_count = BonusFeatureMetrics.clean(raw_column)

        base = FeatureMetrics.from_values(clean_column)
        missing_ratio = missing_count / len(raw_column) * 100
        groups = BonusFeatureMetrics.build_groups(dataframe, feature_name)
        return cls(
            count=base.count,
            mean=base.mean,
            std=base.std,
            min=base.min,
            q1=base.q1,
            q2=base.q2,
            q3=base.q3,
            max=base.max,

            variance=Maths.variance(clean_column),
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            group_stds=Maths.group_stds(groups),
            group_means=Maths.group_means(groups),
            separation_score=Maths.separation_score(groups),
        )

    @classmethod
    def clean(_, raw_column: list[str]) -> tuple[list[float], int] | None:

        missing_count = 0
        clean = []

        for raw in raw_column:
            if CsvManip.is_missing(raw):
                missing_count += 1
                continue
            try:
                clean.append(float(raw))
            except Exception:
                return None
        return clean, missing_count

    @classmethod
    def build_groups(
            _,
            dataframe: pandas.DataFrame,
            feature_name: str
                ) -> dict[str, list[float]]:
        house_col = "Hogwarts House"

        houses = dataframe[house_col].unique()

        groups: dict[str, list[float]] = {house: [] for house in houses}

        feature_values = dataframe[feature_name].tolist()
        house_values = dataframe[house_col].tolist()

        for house, raw in zip(house_values, feature_values):

            if CsvManip.is_missing(raw):
                continue

            try:
                value = float(raw)
            except Exception:
                continue

            groups[house].append(value)

        return groups


@dataclass(slots=True)
class BonusReport(DescribeReport):
    by_bonus: dict[str, BonusFeatureMetrics]
    by_feature: dict[str, FeatureMetrics]

    @classmethod
    def from_dataframe(cls, dataframe: pandas.DataFrame) -> "BonusReport":

        features = CsvManip.loadFeatures(dataframe)

        by_feature: dict[str, FeatureMetrics] = {}
        by_bonus: dict[str, BonusFeatureMetrics] = {}

        for feature_name, values in features.items():

            base_metrics = FeatureMetrics.from_values(values)
            bonus_metrics = BonusFeatureMetrics.from_dataframe(
                dataframe,
                feature_name,
            )
            by_feature[feature_name] = base_metrics
            by_bonus[feature_name] = bonus_metrics

        return cls(
            by_feature=by_feature,
            by_bonus=by_bonus,
        )

    def print_bonus(self) -> None:
        if not self.by_bonus:
            print("No bonus metrics available.")
            return

        metric_names = list(BONUS_LABELS.keys())

        metric_label_width = max(len(BONUS_LABELS[m]) for m in metric_names)

        terminal_width = self._terminal_width()
        columns = self._build_feature_columns()

        blocks = self._split_columns_to_fit_terminal(
            columns,
            terminal_width=terminal_width,
            metric_label_width=metric_label_width,
        )

        lines: list[str] = []

        for block_index, block in enumerate(blocks):

            if block_index > 0:
                lines.append("")

            left_padding = " " * (metric_label_width + 1)
            headers = " ".join(
                f"{column.header:>{column.width}}" for column in block
            )
            lines.append(left_padding + headers)

            for metric_name in metric_names:

                label = BONUS_LABELS[metric_name]

                parts = [f"{label:<{metric_label_width}}"]

                for column in block:

                    metrics = self.by_bonus[column.feature_name]
                    value = getattr(metrics, metric_name)

                    parts.append(self._format_number(value, column.width))

                lines.append(" ".join(parts))

        print("\n".join(lines))

    def print_feature_ranking(self) -> None:
        print("\nFeature Ranking\n")

        rows = []

        for feature, metrics in self.by_bonus.items():
            rows.append((
                feature,
                metrics.separation_score,
                metrics.missing_ratio))

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

    def print_feature_insights(self) -> None:

        print("\nFeature Insights\n")

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
                print("  Insight  : distribution almost identical between houses.")
            elif sep < 3:
                print("  Insight  : partial separation but strong overlaps.")
            else:
                print("  Insight  : clear clustering between houses.")

            print()
    
    def print_dataset_diagnostics(self) -> None:

        print("\nDataset Diagnostics\n")

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

    def print_summary(self) -> None:
        self.print_feature_ranking()
        self.print_feature_insights()
        self.print_dataset_diagnostics()


def filter_dataframe(
    dataframe: pandas.DataFrame,
    houses: list[str] | None,
    feature_names: list[str] | None,
) -> pandas.DataFrame:
    filtered = dataframe.copy()
    house_col = "Hogwarts House"

    if houses is not None and house_col in filtered.columns:
        filtered = filtered[filtered[house_col].isin(houses)]

    if feature_names is not None:
        kept_columns: list[str] = []

        if house_col in filtered.columns:
            kept_columns.append(house_col)

        for name in feature_names:
            if name in filtered.columns and name not in kept_columns:
                kept_columns.append(name)

        filtered = filtered[kept_columns]

    return filtered


def parse_args(argv: list[str] | None = None):
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


def main(argv: list[str] | None = None) -> int:
    path, houses, feature_names, summary, base = parse_args(argv)

    try:
        dataframe = CsvManip.loadCsv(path)
    except Exception as exc:
        print(f"Cannot load CSV: {exc!s}")
        return 1

    filtered = filter_dataframe(dataframe, houses, feature_names)

    report = BonusReport.from_dataframe(filtered)

    if base:
        print("==== Base Report ====")
        print(report)

    print("==== Bonus Report ====")
    report.print_bonus()

    if summary:
        print("==== Summary ====")
        report.print_summary()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
