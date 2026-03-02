import math
import pandas
import shutil
import sys
from dataclasses import dataclass, fields

from maths import our_mean, our_std, our_quartile, our_min_max
from shared import load, is_missing

LABELS = {
    "count": "Count",
    "mean":  "Mean",
    "std":   "Std",
    "min":   "Min",
    "q1":   "25%",
    "q2":   "50%",
    "q3":   "75%",
    "max":   "Max",
}
PRECISION = 6        # rounded float
MAX_HEADER_W = 18    # truncate long feature names in header


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
    def from_dataframe(cls, df) -> "DescribeReport":
        by: dict[str, FeatureMetrics] = {}
        for col in df.columns:
            if col.strip().lower() == "index":
                continue
            vals: list[float] = []
            numeric = True
            for raw in df[col].tolist():
                if is_missing(raw):
                    continue
                try:
                    vals.append(float(raw))
                except Exception:
                    numeric = False
                    break
            if numeric and vals:
                by[col] = FeatureMetrics.from_values(vals)
        return cls(by)

    def _short(name: str) -> str:
        if len(name) <= MAX_HEADER_W:
            return name
        return name[:MAX_HEADER_W - 3] + "..."

    def _metric_field_names(self) -> list[str]:
        return [f.name for f in fields(FeatureMetrics)]

    def __str__(self) -> str:
        cols = list(self.by_feature.keys())
        if not cols:
            return "No numeric features found."
        metric_names = self._metric_field_names()
        term_w = shutil.get_terminal_size((120, 20)).columns
        # label width
        label_w = max(len(LABELS.get(m, m)) for m in metric_names)
        # display header + column width
        disp = [DescribeReport._short(c) for c in cols]
        col_ws = [max(12, len(d)) for d in disp]
        # chunk columns to fit terminal width
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
                lines.append("")  # blank line between blocks
            # header/Features Names
            header = " " * (label_w + 1) +\
                " ".join(f"{d:>{w}}" for _, d, w in block)
            lines.append(header)
            # rows
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

    df = load(av[1])
    if not isinstance(df, pandas.DataFrame):
        print("Error: Cannot load CSV file")
        return 1

    report = DescribeReport.from_dataframe(df)
    print(report)
#   JUST HERE TO SEE THE DIFF
#   print(df.describe())
    return 0


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
