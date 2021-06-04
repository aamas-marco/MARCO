
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
# mf env
from envs.switch_env_multi import SwitchGame as SwitchGame
from envs.switch_env_bridge import SwitchGame as SwitchGameWithBridge

# mb env
from envs.mb_switch_env_multi import SwitchWithModel

class EpisodeRunner:

    # if enviornment is batched, then episode runner only handels logging of test stats
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = 1 # always 1
        # self.env = SwitchWithModel(n_agents=args.n_agents, env_path=args.enviornment_model_directory,
        #                            reward_type=args.reward_type, ensemble=args.ensemble, beta1=self.args.beta1,
        #                               beta3=args.beta3, exploration=False)
        self.mbenv = None
        if args.n_bridges > 0:
            self.mfenv = SwitchGameWithBridge(n_agents=self.args.n_agents, n_switches=self.args.n_switches,
                                              n_bridges=args.n_bridges, episode_limit=self.args.episode_limit)
        else:
            self.mfenv = SwitchGame(n_agents=self.args.n_agents, n_switches=self.args.n_switches,
                                              episode_limit=self.args.episode_limit)

        self.env = self.mfenv
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        self.max_returns_test = []
        self.max_returns = []

        # Log the first run
        self.log_train_stats_t = -1000000
        self.true_t_env = 0

    def reset_env(self,  mb=True):
        try:
            self.close_env()
        except:
            print ("no env to close")
        if mb:
            self.env = self.mbenv
        else:
            self.env = self.mfenv
        self.reset()

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
            # print (e)
            pass

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, train_runner_t_env=None, batched=False):

        if batched:
            self.true_t_env = train_runner_t_env
        else:
            self.true_t_env = self.t_env
        self.reset()
        terminated = False
        episode_return = 0
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
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.true_t_env, test_mode=test_mode)
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

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
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.true_t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        if test_mode:
            self.max_returns_test.append(self.env.max_reward)
        else:
            self.max_returns.append(self.env.max_reward)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            print ("logging because of 1")
            self._log(cur_returns, cur_stats, log_prefix)
        # logging of training progress should only be handeled by parallel runenr if env is batched
        elif (self.true_t_env - self.log_train_stats_t >= self.args.runner_log_interval) and batched is False:
            print("logging because of 2", log_prefix)
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.true_t_env)
            self.log_train_stats_t = self.true_t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        # move return from [-1,1] to [1,2]

        if prefix == 'test_':
            assert len(returns) == len(self.max_returns_test)
            return_normalized = (np.mean(returns) + 1) / (np.mean(self.max_returns_test) + 1 + 1e-8)
            self.max_returns_test.clear()
        else:
            assert len(returns) == len(self.max_returns)
            return_normalized = (np.mean(returns) + 1) / (np.mean(self.max_returns) + 1 + 1e-8)
            self.max_returns.clear()

        self.logger.log_stat(prefix + "return_mean", return_normalized, self.true_t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.true_t_env)

        print ("true_t_env:",self.true_t_env, "test return normalized: ", return_normalized)
        # clear
        returns.clear()


        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.true_t_env)
        stats.clear()
