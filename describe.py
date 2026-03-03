import math
import shutil
import sys
from dataclasses import dataclass, fields

from maths import our_mean, our_std, our_quartile, our_min_max
from CsvManip import CsvManip


LABELS = {
    "count": "Count",
    "mean":  "Mean",
    "std":   "Std",
    "min":   "Min",
    "q1":    "25%",
    "q2":    "50%",
    "q3":    "75%",
    "max":   "Max",
}
PRECISION = 6
MAX_HEADER_W = 18


@dataclass(slots=True)
class FeatureMetrics:
    count: float
    mean: float
    std: float
    min: float
    q1: float
    q2: float
    q3: float
    max: float

    @classmethod
    def from_values(cls, xs: list[float]) -> "FeatureMetrics":
        n = len(xs)
        if n == 0:
            nan = math.nan
            return cls(nan, nan, nan, nan, nan, nan, nan, nan)

        mn, mx = our_min_max(xs)
        return cls(
            count=float(n),
            mean=our_mean(xs),
            std=our_std(xs),
            min=mn,
            q1=our_quartile(xs, 0.25),
            q2=our_quartile(xs, 0.50),
            q3=our_quartile(xs, 0.75),
            max=mx,
        )


@dataclass(slots=True)
class DescribeReport:
    by_feature: dict[str, FeatureMetrics]

    @classmethod
    def from_features(cls,
                      features: dict[str, list[float]]) -> "DescribeReport":
        by: dict[str, FeatureMetrics] = {}
        for name, values in features.items():
            if values:
                by[name] = FeatureMetrics.from_values(values)
        return cls(by)

    @classmethod
    def from_csv(cls, path: str) -> "DescribeReport":
        csv = CsvManip(path)
        return cls.from_features(csv.features)

    @staticmethod
    def _short(name: str) -> str:
        if len(name) <= MAX_HEADER_W:
            return name
        return name[: MAX_HEADER_W - 3] + "..."

    def _metric_field_names(self) -> list[str]:
        return [f.name for f in fields(FeatureMetrics)]

    def __str__(self) -> str:
        cols = list(self.by_feature.keys())
        if not cols:
            return "No numeric features found."

        metric_names = self._metric_field_names()
        term_w = shutil.get_terminal_size((120, 20)).columns

        label_w = max(len(LABELS.get(m, m)) for m in metric_names)

        disp = [DescribeReport._short(c) for c in cols]
        col_ws = [max(12, len(d)) for d in disp]

        chunks = []
        cur = []
        used = label_w + 1

        for c, d, w in zip(cols, disp, col_ws):
            need = w + 1
            if cur and used + need > term_w:
                chunks.append(cur)
                cur = []
                used = label_w + 1
            cur.append((c, d, w))
            used += need

        if cur:
            chunks.append(cur)

        lines: list[str] = []
        for block_i, block in enumerate(chunks):
            if block_i > 0:
                lines.append("")

            header = " " * (label_w + 1) +\
                " ".join(f"{d:>{w}}" for _, d, w in block)
            lines.append(header)

            for m in metric_names:
                label = LABELS.get(m, m)
                line = f"{label:<{label_w}} "
                for feat, _, w in block:
                    v = getattr(self.by_feature[feat], m)
                    if isinstance(v, float) and math.isnan(v):
                        cell = f"{'nan':>{w}}"
                    else:
                        cell = f"{v:>{w}.{PRECISION}f}"
                    line += cell + " "
                lines.append(line.rstrip())

        return "\n".join(lines)


def main(ac: int, av: list[str]) -> int:
    if ac != 2:
        print("Error: Wrong numbers of argument")
        return 1
    dataframe = CsvManip.loadCsv(av[1])
    features = CsvManip.loadFeatures(
        dataframe,
        ignore_cols={'index'}
    )
    report = DescribeReport.from_features(features)
    print(report)
    # debug comparison
    # print(csv.dataframe.describe())
    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)