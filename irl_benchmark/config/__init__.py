"""This module contains configurations for IRL and RL algorithms."""

from copy import copy

# A dictionary containing allowed and default values for
# the config of each IRL algorithm.
# To be extended in each specific IRL algorithm implementation:
IRL_CONFIG_DOMAINS = {}

# A dictionary containing allowed and default values for
# the config of each RL algorithm.
# To be extended in each specific RL algorithm implementation:
RL_CONFIG_DOMAINS = {}


def preprocess_config(config_for: object, domains: dict, config: dict) -> dict:
    """ Pre-processes a config dictionary.

    This is based on the values specified for each IRL algorithm in
    IRL_CONFIG_DOMAINS. The following steps are performed:

    * If values in config are missing, add their default values.
    * If values are illegal (e.g. too high), raise an error.
    * If unknown fields are specified, raise an error.

    Manipulates a copy of the passed config and returns it.

    Parameters
    ----------
    config: dict
        The unprocessed config dictionary.

    Returns
    -------
    dict
        The processed config dictionary.

    """
    # replace config by empty dictionary if None:
    if config is None:
        config = {}
    else:
        config = copy(config)

    # get config domain for the correct subclass calling this:
    config_domain: dict = domains[type(config_for)]
    for key in config_domain.keys():
        if key in config.keys():
            # for numerical fields:
            if config[key] is None and 'optional' in config_domain[key].keys() \
                    and config_domain[key]['optional']:
                # encountered optional field with value None,
                # no checks necessary
                continue
            elif config_domain[key]['type'] in [float, int]:
                # check if right type:
                assert isinstance(
                    config[key], config_domain[key]
                    ['type']), "Wrong config value type for key " + str(key)
                # check if value is high enough:
                assert config[key] >= config_domain[key][
                    'min'], "Config value too low for key " + str(key)
                # check if value is low enough:
                assert config[key] <= config_domain[key][
                    'max'], "Config value too high for key " + str(key)
            # for categorical fields:
            elif config_domain[key]['type'] == 'categorical':
                # check if value is allowed:
                assert config[key] in config_domain[key][
                    'values'], "Illegal config value : " + config[key]
            elif config_domain[key]['type'] is bool:
                assert isinstance(config[key], bool)
            else:
                # encountered type for which no implementation has been written
                # extend code above to fix.
                raise NotImplementedError(
                    "No implementation for config value type: " +
                    str(config_domain[key]['type']))
        else:
            # key not specified in given config, use default value:
            config[key] = config_domain[key]['default']
    # check if config only contains legal fields:
    for key in config.keys():
        if key not in config_domain.keys():
            raise ValueError("Unknown config field: " + str(key))

    # return the pre-processed config:
    return config
