import numpy as np


def get_nearest_value(
    data,
    value,
):
    """Get the nearest value in a dataset

    Args:
        data: array of values to hunt through
        value: value to find nearest in data to

    Returns:
        Nearest value
    """

    # Find the nearest below and above
    diff = data - value
    less = np.where(diff <= 0)
    greater = np.where(diff >= 0)

    # Get the position and value for above and below. If we're at the edge but somehow this doesn't work, just
    # go for the lowest or highest values
    if len(less[0]) == 0:
        nearest_below = data[0]
    else:
        nearest_below_idx = np.argsort(np.abs(diff[less]))[0]
        nearest_below = data[less][nearest_below_idx]

    if len(greater[0]) == 0:
        nearest_above = data[-1]
    else:
        nearest_above_idx = np.argsort(diff[greater])[0]
        nearest_above = data[greater][nearest_above_idx]

    nearest_value = np.array([nearest_below, nearest_above])

    # If we're actually choosing a grid value, just return that singular value
    if np.diff(nearest_value) == 0:
        nearest_value = nearest_value[0]

    return nearest_value


def get_nearest_values(
    dataset,
    keys,
    values,
):
    """Function to parallelise get_nearest_value

    Args:
        dataset: Dataset to hunt through
        keys (list): List of keys to search with
        values (list): List of values to find in the dataset

    Returns
        list of the nearest values
    """

    nearest_values_list = [
        get_nearest_value(dataset[keys[i]].values, values[i])
        for i in range(len(values))
    ]

    return nearest_values_list
