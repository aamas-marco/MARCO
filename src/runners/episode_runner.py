from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from envs.mpe_env import make_env
import math

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = 1
        self.mbenv = None
        self.mfenv = self.mfenv = make_env("simple_reference")
        self.env = self.mfenv

        self.episode_limit = 25
        self.t = 0
        self.t_env = 0

        # logging
        self.train_returns = []
        self.test_mf_returns = []
        self.test_mb_returns = []
        self.train_stats = {}
        self.test_mf_stats = {}
        self.test_mb_stats = {}

        self.cur_return_test_mf = 0

        # Log the first run
        self.log_train_stats_t = -1000000

        self.no_action_avail = False


    def reset_env(self,  mb=True):
        if mb:
            self.env = self.mbenv
        else:
            self.env = self.mfenv


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        try:
            self.mfenv.close()
            self.mbenv.close()
            self.env.close()
        except Exception as e:
            print (e)
            pass

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, test_mode_mb=False):
        self.reset()
        terminated = False
        episode_return = 0
        episode_return_reward = 0
        episode_return_penalty = 0

        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            if self.args.ensemble > 1:
                if (not test_mode) or (test_mode and test_mode_mb):
                    episode_return_reward += env_info["reward_env"]
                    episode_return_penalty += env_info["reward_tilde"]

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }


            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        if test_mode:
            if test_mode_mb:
                cur_stats = self.test_mb_stats
                cur_returns = self.test_mb_returns
                log_prefix = "testmb_"
            else:
                cur_stats = self.test_mf_stats
                cur_returns = self.test_mf_returns
                log_prefix = "testmf_"
        else:
            cur_stats = self.train_stats
            cur_returns = self.train_returns
            log_prefix = ""


        keys = set(cur_stats) | set(env_info)
        if self.args.ensemble > 1:
            keys.discard("reward_env")
            keys.discard("reward_tilde")

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in keys})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if self.args.ensemble > 1:
            if (not test_mode) or (test_mode and test_mode_mb):
                cur_stats["baseline"] = self.env.baseline

        if (not test_mode) or (test_mode and test_mode_mb):
            cur_stats["reward_env"] = episode_return_reward + cur_stats.get("reward_env", 0)
            cur_stats["reward_tilde"] = episode_return_penalty + cur_stats.get("reward_tilde", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and not test_mode_mb and (len(self.test_mf_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif test_mode and test_mode_mb and (len(self.test_mb_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        if prefix == "testmf_":
            self.cur_return_test_mf = np.mean(returns)
            if math.isnan(self.cur_return_test_mf):
                raise ValueError("nan current return")

        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes" and k != 'reward_':
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()