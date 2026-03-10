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
        select_cols: set[str] | None = None
    ) -> tuple[dict[str, list[float]], dict[str, tuple[int, int]]]:
        """
        Extract numeric features from a DataFrame.

        Args:
            dataframe: Source pandas DataFrame.
            houses:Keeps only rows where column "Hogwarts House" equals houses.
            ignore_cols: Column names to ignore , default to {"index"}.

        Returns:
            Tuple : dict[str, list[float]] , dict[str, tuple[int, int]]
            Dict mapping feature name -> list of float values.
            Dict mapping feature name -> count missing , count all


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
        missing: dict[str, tuple[int, int]] = {}

        for feature_name in filtered.columns:
            if feature_name.strip().lower() in ignore:
                continue
            if not select_cols or feature_name.strip().lower() in select_cols:
                values: list[float] = []
                missing_count = 0
                numeric = True

                for raw in filtered[feature_name].tolist():
                    if CsvManip.is_missing(raw):
                        missing_count += 1
                        continue
                    try:
                        values.append(float(raw))
                    except Exception:
                        numeric = False
                        break

                if numeric and values:
                    total = len(filtered[feature_name])
                    features[feature_name] = values
                    missing[feature_name] = missing_count, total

        return features, missing

    @staticmethod
    def loadFeaturesMatrix(
        dataframe: pandas.DataFrame,
        feature_names: list[str],
        *,
        houses: set[str] | None = None,
        labels: bool = False,
    ) -> tuple[list[str], list[list[float]], list[str] | None]:
        """
        Extract a numeric feature matrix from a DataFrame.

        Args:
            dataframe:
                Source pandas DataFrame.

            feature_names:
                Ordered list of candidate feature columns to include.

            houses:
                Optional set of house names used to filter rows before
                matrix extraction. Only rows whose "Hogwarts House"
                belongs to this set are kept.

            labels:
                If True, also return the house label for each valid row.

        Returns:
            Tuple containing:
            - names:
                List of feature names actually found in the DataFrame,
                preserving the requested order.
            - matrix:
                List of valid numeric rows. A row is kept only if all
                selected feature values are non-missing and float-convertible.
            - output_labels:
                List of house labels aligned with `matrix` if `labels=True`,
                otherwise None.

        Rules:
            - Rows containing at least one missing or invalid value in the
            selected features are dropped entirely.
            - Feature order in the output matrix matches `names`.
            - If `labels=True` but the house column does not exist,
            empty strings are returned as labels for valid rows.

        Notes:
            This helper is intended for model training and prediction,
            where complete numeric rows are required.
        """
        filtered = dataframe

        house_col = "Hogwarts House"
        if houses is not None and house_col in filtered.columns:
            filtered = filtered[filtered[house_col].isin(houses)]

        names: list[str] = []
        # filter names to only keep numeric columns
        for name in feature_names:
            if name not in filtered.columns or name == "Index":
                continue

            column_ok = True
            for raw_value in filtered[name].tolist():
                if CsvManip.is_missing(raw_value):
                    continue
                try:
                    float(raw_value)
                except Exception:
                    column_ok = False
                    break

            if column_ok:
                names.append(name)

        matrix: list[list[float]] = []
        output_labels: list[str] | None = [] if labels else None

        selected = filtered[names]

        label_values = None
        if labels and house_col in filtered.columns:
            label_values = filtered[house_col].tolist()

        for row_index, row_tuple in enumerate(selected.itertuples(
                                                index=False, name=None
                                                )):
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

    @staticmethod
    def build_groups(
            dataframe: pandas.DataFrame,
            houses: set[str] | None,
            features_names: set[str] | None
                ) -> dict[str, dict[str, list[float]]]:
        """
    Group numeric feature values by house.

    Args:
        dataframe:
            Source pandas DataFrame.

        houses:
            Optional set of houses to include in the grouped output.
            If None, all houses present in the DataFrame are used.

        features_names:
            Optional set of feature names to group.
            If None, all DataFrame columns are considered.

    Returns:
        Nested mapping of the form:
            {
                feature_name: {
                    house_name: [float values, ...],
                    ...
                },
                ...
            }

        For each feature, values are grouped by house and converted
        to float when possible.

    Rules:
        - Missing values are ignored.
        - Non-numeric values are ignored for that feature.
        - The "Index" column is skipped.
        - Every selected house appears in each feature group,
          even if its value list is empty.

    Notes:
        This helper is mainly used by bonus analysis code to compute
        per-house means, per-house standard deviations, separation
        scores, and other class-aware feature metrics.
    """

        house_col = "Hogwarts House"
        if houses is None:
            houses = set(dataframe[house_col].unique())
        if features_names is None:
            features_names = set(dataframe.columns)

        feature_grouped: dict[str, dict[str, list[float]]] = {}

        for name in features_names:
            if name == "Index":
                continue
            groups: dict[str, list[float]] = {house: [] for house in houses}
            feature_values = dataframe[name].tolist()
            house_values = dataframe[house_col].tolist()

            for house, raw in zip(house_values, feature_values):

                if CsvManip.is_missing(raw):
                    continue

                try:
                    value = float(raw)
                except Exception:
                    continue

                groups[house].append(value)

            feature_grouped[name] = groups

        return feature_grouped
