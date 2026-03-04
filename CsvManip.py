"""
CsvManip.py

CSV loading and numeric feature extraction utilities.

The main abstraction of the project is a "feature":
- A feature corresponds to a dataset column that contains numeric values.
- Missing values are ignored.
- A column is numeric if all non-missing values can be converted to float.
"""

import math
import pandas


class CsvManip:
    """
    Csv utilities for this project.

    This class is used as a namespace for static methods.
    The recommended usage is:
        dataframe = CsvManip.loadCsv(path)
        features = CsvManip.loadFeatures(dataframe, ignore_cols={"index"})
    """
    csv_path: str

    @staticmethod
    def loadCsv(path: str) -> pandas.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            path: Path to a CSV file.

        Returns:
            pandas.DataFrame with the CSV contents.

        Notes:
            Returns None if `path` is not a string.
        """
        if not isinstance(path, str):
            return None
        return pandas.read_csv(path)

    @staticmethod
    def is_missing(value) -> bool:
        """
        Return True if `value` should be treated as missing.

        Missing values:
        - None
        - NaN
        - empty or whitespace-only strings
        """
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        return str(value).strip() == ""

    @staticmethod
    def loadFeatures(
        dataframe: pandas.DataFrame,
        *,
        house: str | None = None,
        ignore_cols: set[str] | None = None,
    ) -> dict[str, list[float]]:
        """
        Extract numeric features from a DataFrame.

        Args:
            dataframe: Source pandas DataFrame.
            house:Keeps only rows where column "Hogwarts House" equals `house`.
            ignore_cols: Column names to ignore , default to {"index"}.

        Returns:
            Dict mapping feature name -> list of float values.

        Rules:
            - Missing values are ignored.
            - Included only if all non-missing values are float-convertible.
        """
        filtered = dataframe

        if house is not None:
            house_col = "Hogwarts House"
            if house_col in filtered.columns:
                filtered = filtered[filtered[house_col] == house]

        ignore = {c.strip().lower() for c in (ignore_cols or {"index"})}

        features: dict[str, list[float]] = {}

        for feature_name in filtered.columns:
            if feature_name.strip().lower() in ignore:
                continue

            values: list[float] = []
            numeric = True

            for raw in filtered[feature_name].tolist():
                if CsvManip.is_missing(raw):
                    continue
                try:
                    values.append(float(raw))
                except Exception:
                    numeric = False
                    break

            if numeric and values:
                features[feature_name] = values

        return features
