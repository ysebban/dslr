import pandas
from dataclasses import dataclass, fields


@dataclass(slots=True)
class CsvManip:

    csvPath: str
    features: dict[str, list[float]] = fields(init=False)
    dataframe: pandas.DataFrame = fields(init=False)

    @classmethod
    def __post_init__(self) -> None:
        self.dataframe = self.load_csv(self.csvPath)
        self.features = self.load_features(self.dataframe)

    def load_csv(path: str) -> pandas.DataFrame:
        return

    def loadNumColumn():
        return

    # Return str HOUSE for a given entrie ?
    def getHousePerEntry():
        return
