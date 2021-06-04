import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run_mpe import run
from run_mpe_mb import run as run_mb
from run_mpe_explore import run as run_explore

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "mpe_results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    if config["mb"] == 1:
        run_mb(_run, config, _log)
    elif config["mb"] == 2:
        run_explore(_run, config, _log)
    elif config["mb"] == 0:
        run(_run, config, _log)
    else:
        raise NotImplementedError


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # for _i, _v in enumerate(params):
    #     if _v.split("=")[0] == "--model":
    #         model = _v.split("=")[1]
    #         config_dict["enviornment_model_directory"] = model
    #         del params[_i]
    #         break

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--sampling_timesteps":
            timesteps = int(_v.split("=")[1])
            config_dict["sampling_timesteps"] = timesteps
            # config_dict["max_samples"] = timesteps
            del params[_i]
            break

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--max_samples":
            max_samples = int(_v.split("=")[1])
            config_dict["max_samples"] = max_samples
            del params[_i]
            break

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--ensemble":
            ensemble = int(_v.split("=")[1])
            config_dict["ensemble"] = ensemble
            del params[_i]
            break

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--mb":
            mb = int(_v.split("=")[1])
            config_dict["mb"] = mb
            del params[_i]
            break

    # for _i, _v in enumerate(params):
    #     if _v.split("=")[0] == "--centralized":
    #         centralized = int(_v.split("=")[1])
    #         config_dict["centralized"] = centralized
    #         del params[_i]
    #         break

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--beta3":
            beta3 = int(_v.split("=")[1])
            config_dict["beta3"] = beta3
            del params[_i]
            break

    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--id":
            id = int(_v.split("=")[1])
            config_dict["id"] = id
            del params[_i]
            break

    config_dict["bridge"] = 0
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == "--bridge":
            bridge = int(_v.split("=")[1])
            config_dict["bridge"] = bridge
            del params[_i]
            break

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results_mpe/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)