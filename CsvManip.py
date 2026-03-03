import pandas
import math
from dataclasses import dataclass, field


@dataclass(slots=True)
class CsvManip:
    """
    To be used with obj = new CsvManip(csv_path)
    Or CsvManip.function(arg)
    obj.dataframe -> "raw" panda DataFrame Type
    obj.features -> dict [course_name to course_score]
    """
    csv_path: str

    @staticmethod
    def loadCsv(path: str) -> pandas.DataFrame:

        if not isinstance(path, str):
            return None

        out = pandas.read_csv(path)
        return out

    @staticmethod
    def is_missing(x) -> bool:
        if x is None:
            return True
        if isinstance(x, float) and math.isnan(x):
            return True
        s = str(x).strip()
        return s == ""

    @staticmethod
    def loadFeatures(
        dataframe: pandas.DataFrame,
        *,
        ignore_cols: set[str] | None = None,
    ) -> dict[str, list[float]]:
        """
        Call:obj.extract_numeric_features(data,ignore_cols={'index','stuf'...})
        Reusable logic:
        - skip ignored columns (case-insensitive + strip)
        - ignore missing values
        - keep column only if *all* non-missing values are float-convertible
        """
        ignore = {c.strip().lower() for c in (ignore_cols or {"index"})}

        out: dict[str, list[float]] = {}
        for feature in dataframe.columns:
            if feature.strip().lower() in ignore:
                continue

            values: list[float] = []
            numeric = True

            for raw in dataframe[feature].tolist():
                if CsvManip.is_missing(raw):
                    continue
                try:
                    values.append(float(raw))
                except Exception:
                    numeric = False
                    break

            if numeric and values:
                out[feature] = values

        return out
