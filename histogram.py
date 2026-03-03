import CsvManip
from maths import (our_mean, our_std)


def plot_histogram(
        Ravenclaw: dict[str, list[float]],
        Slytherin: dict[str, list[float]],
        Gryffindor: dict[str, list[float]],
        Hufflepuff: dict[str, list[float]],
        feature_name: str
        ):
    Ravenclaw_mean = our_mean(Ravenclaw[feature_name])
    Slytherin_mean = our_mean(Slytherin[feature_name])
    Gryffindor_mean = our_mean(Gryffindor[feature_name])
    Hufflepuff_mean = our_mean(Hufflepuff[feature_name])

    Ravenclaw_std = our_std(Ravenclaw[feature_name])
    Slytherin_std = our_std(Slytherin[feature_name])
    Gryffindor_std = our_std(Gryffindor[feature_name])
    Hufflepuff_std = our_std(Hufflepuff[feature_name])
    return


def main(ac: int, av: list[str]) -> int:

    if ac != 2:
        print("Error: Wrong numbers of arguments")
        return 1

    dataframe = CsvManip.loadCsv(av[1])
    All_features = CsvManip.loadFeatures(dataframe)
    Ravenclaw_features = CsvManip.loadFeatures(dataframe,
                                               house="Ravenclaw"
                                               )
    Slytherin_features = CsvManip.loadFeatures(dataframe,
                                               house="Slytherin"
                                               )
    Gryffindor_features = CsvManip.loadFeatures(dataframe,
                                                house="Gryffindor"
                                                )
    Hufflepuff_features = CsvManip.loadFeatures(dataframe,
                                                house="Hufflepuff"
                                                )

    for feature_name in All_features.keys():
        plot_histogram(Ravenclaw_features,
                       Slytherin_features,
                       Gryffindor_features,
                       Hufflepuff_features,
                       feature_name)
    return 0


# if __name__ == '__main__':
#     main(len(sys.argv), sys.argv)
