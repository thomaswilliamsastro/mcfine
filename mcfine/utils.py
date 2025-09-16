import os
import pickle
import tomllib

DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DEFAULT_PATH = os.path.join(
    DEFAULT_DIR,
    "toml",
    "config_defaults.toml",
)
LOCAL_DEFAULT_PATH = os.path.join(
    DEFAULT_DIR,
    "toml",
    "local_defaults.toml",
)


def print_config_params():
    """Print out all config parameters, and default values"""

    with open(CONFIG_DEFAULT_PATH, "rb") as f:
        config_defaults = tomllib.load(f)
    with open(LOCAL_DEFAULT_PATH, "rb") as f:
        local_defaults = tomllib.load(f)

    print("Config Defaults:\n")

    for table in config_defaults.keys():

        print(f"[{table}]")

        for key in config_defaults[table].keys():

            value = config_defaults[table][key]
            value_type = type(value)

            print(f"--{key}\n----default: {value}, type: {value_type}")

        print("\n")

    print("Local Defaults:\n")

    for table in local_defaults.keys():

        print(f"[{table}]")

        for key in local_defaults[table].keys():
            value = local_defaults[table][key]
            value_type = type(value)

            print(f"--{key}\n----default: {value}, type: {value_type}")

        print("\n")


def get_dict_val(
    val_dict,
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
        logger.info(f"No {key} provided. Defaulting to {dict_val}")

    return dict_val


def check_overwrite(
    val_dict,
    key,
):
    """Check if overwrite is in a dict.

    Args:
        val_dict (dict): Dictionary of overwrites.
        key (str): Key to check.

    Returns:
        Bool of whether to overwrite or not.

    """

    if "overwrites" in val_dict:

        if key in val_dict["overwrites"]:
            return val_dict["overwrites"][key]
        else:
            return False

    return False


def save_fit_dict(
    fit_dict,
    file_name,
):
    """Save a fit dictionary

    Args:
        fit_dict: fit dictionary to save
        file_name (str): file to save to

    """

    fit_dir = os.path.dirname(file_name)
    if not os.path.exists(fit_dir):
        os.makedirs(fit_dir)

    with open(file_name, "wb") as f:
        pickle.dump(fit_dict, f)


def load_fit_dict(
    file_name,
):
    """Load a pickled fit dict

    Args:
        file_name (str): file name to load

    Returns:
        Fit dictionary

    """

    with open(file_name, "rb") as f:
        fit_dict = pickle.load(f)

    return fit_dict
