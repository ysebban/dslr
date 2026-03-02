import math


def our_mean(feature: list[float]) -> float:

    total_entries = len(feature)
    if total_entries < 2:
        return math.nan

    total_value = sum(feature)
    return total_value / total_entries


def our_std(feature: list[float]) -> float:

    mean = our_mean(feature)
    if mean == math.nan:
        return math.nan

    total_delta_square = 0.0
    for f in feature:
        delta = f - mean
        total_delta_square += delta * delta
    variance = total_delta_square / (len(feature) - 1)
    return math.sqrt(variance)


def our_quartile(feature: list[float], quartile: float) -> float:
    '''
    Quartile -> beetween 0 and 1
    '''
    total_entries = len(feature)
    if total_entries < 2:
        return math.nan

    s_features = sorted(feature)
    res = (total_entries - 1) * quartile
    high = int(math.ceil(res))
    low = int(math.floor(res))

    if high == low:
        return s_features[high]

    delta = s_features[high] - s_features[low]
    weight = res - low
    return s_features[low] + weight * delta


def our_min_max(feature: list) -> tuple[float, float]:

    n = len(feature)
    if n < 2:
        return (math.nan, math.nan)

    min = feature[0]
    max = feature[0]

    for f in feature:
        if f < min:
            min = f
        elif f > max:
            max = f

    return (min, max)
