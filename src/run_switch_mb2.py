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
from runners.episode_runner_switch_mb import EpisodeRunner
from runners.parallel_runner_switch_mb import ParallelRunner
from runners.data_collector import DataCollector

from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

from learn_model import get_new_dataset
from learn_model_switch import train

# from envs.switch_env_multi import SwitchGame
from envs.mb_switch_env_bridge import SwitchWithModel as SwitchWithModelBridge
from envs.mb_switch_env_multi import SwitchWithModel

import torch.multiprocessing as mp

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    args.enviornment_model_directory = args.prefix + "_" + str(args.n_agents)+"a"+str(args.n_switches)+"s"+\
                                           str(args.n_bridges)+"b_ds"+str(args.sampling_timesteps)+"-"+str(args.relearn_interval)+"-" \
                                           +str(args.max_samples) + "_bs"+str(args.batch_size)+"_bsr"+str(args.batch_size_run)\
                                           +"_targetupdate"+ str(args.target_update_interval)\
                                           +"_lr{0:.1E}".format(float(args.lr))+"_epsilonaneal"+ str(args.epsilon_anneal_time)\
                                           + "_ensem"+str(args.ensemble)


    # args.enviornment_model_directory = "test800kbackup"

    # configure tensorboard logger
    unique_token = args.enviornment_model_directory + "_" +\
                   "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "switch_results", "tb_logs")
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

    # train and test runner should share the same policy (i.e. share the same controller)
    train_runner = ParallelRunner(args=args, logger=logger) # collects traj from unbatched switch mf env
    test_runner = EpisodeRunner(args=args, logger=logger) # collects traj from batched switch mb env

    if args.exploration:
        pass # run seperate script for data collection
    else:
        data_collector = DataCollector(args=args, logger=logger)

    # Set up schemes and groups here
    env_info = test_runner.get_env_info()
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
    print(scheme)


    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }


    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    train_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    test_runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)
    data_collector.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # print (scheme)
    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # learn initial models with **random** policy
    dataset, _ = data_collector.run(args.sampling_timesteps, epsilon=1.0)

    get_new_dataset(dataset, t_env=0, sampling_timesteps=args.sampling_timesteps, switch=True,
                    env_model_directory=args.enviornment_model_directory, policy_model_path="", epsilon=args.epsilon,
                    n_agents=args.n_agents)

    processes = []
    for i in range(args.ensemble):
        pargs = ("state_state", 0)
        pkwargs = {"epochs": 700, "learning_rate": 1e-3, "batch_size": 1000, "hidden_size": 500, "debug": True,
                   "env_model_directory": args.enviornment_model_directory, "model_num": i,
                   "n_agents": args.n_agents,
                   "n_switches": args.n_switches, "n_bridges": args.n_bridges, "print_progress": False}
        p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
        processes.append(p)
        p.daemon = True
        p.start()

    for model_type in ["state_obs", "state_done", "state_avail", "state_reward"]:
        pargs = (model_type, 0,)
        pkwargs = {"epochs": 700, "learning_rate": 1e-3, "batch_size": 1000, "hidden_size": 500, "debug": True,
                   "env_model_directory": args.enviornment_model_directory, "model_num": 0, "n_agents": args.n_agents,
                  "n_switches": args.n_switches, "n_bridges": args.n_bridges, "print_progress": False}
        p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
        processes.append(p)
        p.daemon = True
        p.start()

    for p in processes:
        p.join()

    # env batch size specified by batch_size_run
    if args.n_bridges > 0:
        train_runner.env = SwitchWithModelBridge(n_agents=args.n_agents, n_switches=args.n_switches, n_bridges=args.n_bridges,
                                            episode_limit=args.episode_limit, batch_size=args.batch_size_run,
                                            env_path=args.enviornment_model_directory,
                                            reward_type=args.reward_type, ensemble=args.ensemble, beta1=args.beta1,
                                            beta3=args.beta3, exploration=False)
    else:
        train_runner.env = SwitchWithModel(n_agents=args.n_agents, n_switches=args.n_switches,
                                            episode_limit=args.episode_limit, batch_size=args.batch_size_run,
                                            env_path=args.enviornment_model_directory,
                                            reward_type=args.reward_type, ensemble=args.ensemble, beta1=args.beta1,
                                            beta3=args.beta3, exploration=False)


    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    env_model_save_time = 0
    dataset_size = args.sampling_timesteps

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while train_runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        assert train_runner.env.mb == True
        episode_batch = train_runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, train_runner.t_env, episode)


        if args.save_model and (train_runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            print ("*********************** running tests")
            model_save_time = train_runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(train_runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        n_test_runs = max(1, args.test_nepisode // test_runner.batch_size)
        if (train_runner.t_env - last_test_T) / args.test_interval >= 1.0:
            # Execute test runs
            logger.console_logger.info("t_env: {} / {}".format(train_runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, train_runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()
            # run test episodes with mf enviornment
            assert test_runner.env.mf == True
            for _ in range(n_test_runs):
                test_runner.run(test_mode=True, train_runner_t_env=train_runner.t_env, batched=True)

            # update last time we ran test
            last_test_T = train_runner.t_env


        # collect more samples, learn env model
        if train_runner.t_env - env_model_save_time >= args.relearn_interval and dataset_size < args.max_samples:

            dataset_size += args.sampling_timesteps
            env_model_save_time = train_runner.t_env

            dataset, _ = data_collector.run(args.sampling_timesteps, epsilon=args.epsilon)
            get_new_dataset(dataset, t_env=train_runner.t_env, sampling_timesteps=args.sampling_timesteps, switch=True,
                            env_model_directory=args.enviornment_model_directory, policy_model_path="",
                            epsilon=args.epsilon, n_agents=args.n_agents)

            processes = []
            for i in range(args.ensemble):
                pargs = ("state_state", train_runner.t_env)
                pkwargs = {"epochs": 700, "learning_rate": 1e-3, "batch_size": 1000, "hidden_size": 500, "debug": True,
                           "env_model_directory": args.enviornment_model_directory, "model_num": i,
                           "n_agents": args.n_agents,
                           "n_switches": args.n_switches, "n_bridges": args.n_bridges, "print_progress": False}
                p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
                processes.append(p)
                p.daemon = True
                p.start()

            for model_type in [ "state_obs", "state_done", "state_avail", "state_reward"]:
                pargs = (model_type, train_runner.t_env)
                pkwargs = {"epochs": 700, "learning_rate": 1e-3, "batch_size": 1000, "hidden_size": 500, "debug": True,
                           "env_model_directory": args.enviornment_model_directory, "model_num": 0,
                           "n_agents": args.n_agents,
                           "n_switches": args.n_switches, "n_bridges": args.n_bridges, "print_progress": False}
                p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
                processes.append(p)
                p.daemon = True
                p.start()

            for p in processes:
                p.join()

            # refresh mb enviornment with newly learnt dynamics
            train_runner.env.refresh_env()
            assert train_runner.env.mb == True

        episode += args.batch_size_run

        if (train_runner.t_env - last_log_T) >= args.log_interval:
            print ("logged")
            logger.log_stat("episode", episode, train_runner.t_env)
            logger.print_recent_stats()
            last_log_T = train_runner.t_env

    train_runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    # if config["test_nepisode"] < config["batch_size_run"]:
    #     config["test_nepisode"] = config["batch_size_run"]
    # else:
    #     config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config