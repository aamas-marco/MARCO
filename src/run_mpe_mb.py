import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners.episode_runner_mpe_mb import EpisodeRunner
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from runners.data_collector_mpe import DataCollector

from learn_model import get_new_dataset
from learn_model_mpe import train

from envs.mb_mpe_env import MpeWithModel

import torch.multiprocessing as mp

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    # if args.deivce == "cpu":
    #     raise ValueError("cpu")
    tmp = "%s_mb_samplingt%d_maxsamples%d_ensemble%d" \
          % (args.prefix, args.sampling_timesteps, args.max_samples,
             args.ensemble)

    # if args.ensemble > 1:
    #     tmp = tmp + "_beta3" + str(args.beta3)

    # tmp = tmp + ["", "_centralized"][args.centralized]
    # tmp = "test_1b0.0_samplingt1000_relearnint10000_ensemble5_epsilon0.1_cat_nagents3"
    args.enviornment_model_directory = str(args.id) + "_" + tmp
    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = args.enviornment_model_directory + "_" + \
                   "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "mpe_results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    try:
        mp.set_start_method('spawn')
    except:
        pass

    # Init runner so we can get env info
    runner = EpisodeRunner(args=args, logger=logger)
    data_collector = DataCollector(args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess, device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args, exploration=False)
    # mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
     # Give runner the scheme
    data_collector.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()


    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    last_save_T = 0
    last_relearn_T = 0
    dataset_size = args.sampling_timesteps

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    dataset = data_collector.run(args.sampling_timesteps, epsilon=1.0)
    get_new_dataset(dataset, t_env=0, sampling_timesteps=args.sampling_timesteps,
                    env_model_directory=args.enviornment_model_directory, policy_model_path="", epsilon=args.epsilon)

    print ("done collecting dataset")

    # processes = []
    # for i in range(args.ensemble):
    #     pargs = ("state_state", 0,)
    #     pkwargs = {"epochs": 500, "learning_rate": 1e-3, "batch_size": 100, "hidden_size": 500, "debug": False,
    #                "env_model_directory": args.enviornment_model_directory, "model_num": i, "n_agents":args.n_agents}
    #     p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
    #     processes.append(p)
    #     p.daemon = True
    #     p.start()
    # for model_type in ["state_done", "state_reward"]:
    #     pargs = (model_type, 0,)
    #     pkwargs = {"epochs": 500, "learning_rate": 1e-3, "batch_size": 100, "hidden_size": 100, "debug": False,
    #         "env_model_directory": args.enviornment_model_directory, "reward_type": args.reward_type, "model_num": i,
    #         "n_agents":args.n_agents}
    #     p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
    #     processes.append(p)
    #     p.daemon = True
    #     p.start()

    # for p in processes:
    #     p.join()

    for i in range(args.ensemble):
        pargs = ("state_state", 0,)
        pkwargs = {"epochs": 5000, "learning_rate": 1e-3, "batch_size": 500, "hidden_size": 500, "debug": True,
                   "env_model_directory": args.enviornment_model_directory, "model_num": i, "n_agents":args.n_agents}
        train(*pargs, **pkwargs)
        # p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
        # processes.append(p)
        # p.daemon = True
        # p.start()
    for model_type in ["state_reward"]:
        for i in range(args.ensemble):
            pargs = (model_type, 0,)
            pkwargs = {"epochs": 5000, "learning_rate": 1e-3, "batch_size": 500, "hidden_size": 500, "debug": True,
                "env_model_directory": args.enviornment_model_directory, "reward_type": args.reward_type, "model_num": i,
                "n_agents":args.n_agents}
            train(*pargs, **pkwargs)
        # p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
        # processes.append(p)
        # p.daemon = True
        # p.start()

    # for p in processes:
    #     p.join()

    # runner.mbenv = MpeWithModel(env_path=args.enviornment_model_directory,
    #                                 reward_type=args.reward_type, ensemble=args.ensemble, beta1=args.beta1,
    #                                 beta3=args.beta3, exploration=False)

    runner.mbenv = MpeWithModel(env_path=args.enviornment_model_directory,
                                    reward_type=args.reward_type, ensemble=args.ensemble, beta1=args.beta1,
                                    beta3=args.beta3, exploration=args.ensemble>1)
    runner.env = runner.mbenv

    while runner.t_env <= args.t_max:

        #mb
        # Run for a whole episode at a time
        assert runner.env.mb == True
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)
        assert runner.env.mb == True

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            assert runner.env.mb == True
            learner.train(episode_sample, runner.t_env, episode)
            assert runner.env.mb == True

        #mf
        # test
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            # Execute test runs
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            # run test episodes with mf enviornment
            runner.reset_env(mb=False)
            assert runner.env.mf == True
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

            # update last time we ran test
            last_test_T = runner.t_env

            #switch back to mb
            runner.reset_env(mb=True)
            assert runner.env.mb == True

        if args.save_model and (runner.t_env - last_save_T >= 20000):

            last_save_T = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading
            learner.save_models(save_path)

        if (((runner.t_env - last_relearn_T) / args.relearn_interval) >= 1.0) and (dataset_size < args.max_samples):

            last_relearn_T = runner.t_env
            dataset_size += args.sampling_timesteps
            # dataset = data_collector.run(args.sampling_timesteps, epsilon=args.epsilon)
            dataset = data_collector.run(args.sampling_timesteps, epsilon=mac.action_selector.epsilon)
            get_new_dataset(dataset, t_env=runner.t_env, sampling_timesteps=args.sampling_timesteps,
                            env_model_directory=args.enviornment_model_directory,
                            policy_model_path="", epsilon=args.epsilon)

            processes = []
            for i in range(args.ensemble):
                pargs = ("state_state", runner.t_env)
                pkwargs = {"epochs": 5000, "learning_rate": 1e-3, "batch_size": 500, "hidden_size": 500, "debug": False,
                           "env_model_directory": args.enviornment_model_directory, "model_num": i}
                p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
                processes.append(p)
                p.daemon = True
                p.start()
            for model_type in ["state_reward"]:
                for i in range(args.ensemble):
                    pargs = (model_type, runner.t_env)
                    pkwargs = {"epochs": 5000, "learning_rate": 1e-3, "batch_size": 500, "hidden_size": 500, "debug": False,
                            "env_model_directory": args.enviornment_model_directory, "reward_type": args.reward_type,
                            "model_num": i}
                    p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
                    processes.append(p)
                    p.daemon = True
                    p.start()

            for p in processes:
                p.join()

            runner.mbenv.refresh_env()
            runner.reset_env(mb=True)
            assert runner.env.mb == True

        episode += args.batch_size_run


        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config