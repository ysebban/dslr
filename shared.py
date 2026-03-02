import pandas
import math


def load(path: str) -> pandas.DataFrame:
    if not isinstance(path, str):
        return None
    out = pandas.read_csv(path)
    return out


def is_missing(x) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    s = str(x).strip()
    return s == ""
