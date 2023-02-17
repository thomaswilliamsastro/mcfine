import pickle


def get_dict_val(val_dict,
                 default_val_dict,
                 table,
                 key,
                 logger=None,
                 ):
    """Get value from a dictionary, falling to default if not present.

    Args:
        val_dict (dict): Provided dictionary of values
        default_val_dict (dict): Dictionary of fallback values
        table (str): Table key to search for
        key (str): Dictionary key to search for
        logger: Optional logger instance

    Returns:
        Dictionary value
    """

    if table in val_dict:
        if key in val_dict[table]:
            dict_val = val_dict[table][key]
            return dict_val

    dict_val = default_val_dict[table][key]
    if logger is not None:
        logger.info('No %s provided. Defaulting to %s' % (key, dict_val))

    return dict_val


def check_overwrite(val_dict,
                    key,
                    ):
    """Check if overwrite is in a dict.

    Args:
        val_dict (dict): Dictionary of overwrites.
        key (str): Key to check.

    Returns:
        Bool of whether to overwrite or not.

    """

    if 'overwrites' in val_dict:

        if key in val_dict['overwrites']:
            return val_dict['overwrites'][key]
        else:
            return False

    return False


def save_fit_dict(fit_dict,
                  file_name,
                  ):
    """Save a fit dictionary

    Args:
        fit_dict: fit dictionary to save
        file_name (str): file to save to

    """

    with open(file_name, 'wb') as f:
        pickle.dump(fit_dict, f)


def load_fit_dict(file_name,
                  ):
    """Load a pickled fit dict

    Args:
        file_name (str): file name to load

    Returns:
        Fit dictionary

    """

    with open(file_name, 'rb') as f:
        fit_dict = pickle.load(f)

    return fit_dict
