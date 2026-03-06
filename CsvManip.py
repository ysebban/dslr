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
        houses: set[str] | None = None,
        ignore_cols: set[str] | None = None,
    ) -> dict[str, list[float]]:
        """
        Extract numeric features from a DataFrame.

        Args:
            dataframe: Source pandas DataFrame.
            houses:Keeps only rows where column "Hogwarts House" equals houses.
            ignore_cols: Column names to ignore , default to {"index"}.

        Returns:
            Dict mapping feature name -> list of float values.

        Rules:
            - Missing values are ignored.
            - Included only if all non-missing values are float-convertible.
        """
        filtered = dataframe

        if houses is not None:
            house_col = "Hogwarts House"
            if house_col in filtered.columns:
                filtered = filtered[filtered[house_col].isin(houses)]

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

    # @staticmethod
    # def loadFeaturesMatrix(
    #     dataframe: pandas.DataFrame,
    #     feature_names: list[str],
    #     *,
    #     houses: set[str] | None = None,
    #     labels: bool = False,
    # ) -> tuple[list[str], list[list[float]], list[str] | None]:
    #     """
    #     Extract a row-aligned numeric feature matrix from a DataFrame.

    #     Args:
    #         dataframe: Source pandas DataFrame.
    #         feature_names: Selected feature names to keep.
    #         houses: Keeps only rows where column "Hogwarts House" is in houses.
    #         labels: If True, also returns aligned house labels.

    #     Returns:
    #         A tuple containing:
    #             - ordered feature names
    #             - matrix of numeric rows
    #             - list of labels, or None

    #     Rules:
    #         - Selected features must stay aligned by row.
    #         - If one selected value is invalid, the whole row is ignored.
    #         - Feature order is preserved from the DataFrame columns.
    #     """
    #     filtered = dataframe

    #     house_col = "Hogwarts House"
    #     if houses is not None and house_col in filtered.columns:
    #         filtered = filtered[filtered[house_col].isin(houses)]

    #     names = [col for col in filtered.columns if col in feature_names]

    #     matrix: list[list[float]] = []
    #     output_labels: list[str] | None = [] if labels else None

    #     for _, row in filtered.iterrows():
    #         row_values: list[float] = []
    #         valid_row = True

    #         for name in names:
    #             raw_value = row[name]

    #             if CsvManip.is_missing(raw_value):
    #                 valid_row = False
    #                 break

    #             try:
    #                 row_values.append(float(raw_value))
    #             except Exception:
    #                 valid_row = False
    #                 break

    #         if not valid_row:
    #             continue

    #         matrix.append(row_values)

    #         if labels and output_labels is not None:
    #             if house_col in filtered.columns:
    #                 output_labels.append(row[house_col])
    #             else:
    #                 output_labels.append("")

    #     return (names, matrix, output_labels)
    @staticmethod
    def loadFeaturesMatrix(
        dataframe: pandas.DataFrame,
        feature_names: list[str],
        *,
        houses: set[str] | None = None,
        labels: bool = False,
    ) -> tuple[list[str], list[list[float]], list[str] | None]:
        filtered = dataframe

        house_col = "Hogwarts House"
        if houses is not None and house_col in filtered.columns:
            filtered = filtered[filtered[house_col].isin(houses)]

        names = [name for name in feature_names if name in filtered.columns]

        matrix: list[list[float]] = []
        output_labels: list[str] | None = [] if labels else None

        selected = filtered[names]

        label_values = None
        if labels and house_col in filtered.columns:
            label_values = filtered[house_col].tolist()

        for row_index, row_tuple in enumerate(selected.itertuples(index=False, name=None)):
            row_out: list[float] = []
            valid_row = True

            for raw_value in row_tuple:
                if CsvManip.is_missing(raw_value):
                    valid_row = False
                    break
                try:
                    row_out.append(float(raw_value))
                except Exception:
                    valid_row = False
                    break

            if not valid_row:
                continue

            matrix.append(row_out)

            if output_labels is not None:
                if label_values is not None:
                    output_labels.append(label_values[row_index])
                else:
                    output_labels.append("")

        return (names, matrix, output_labels)
