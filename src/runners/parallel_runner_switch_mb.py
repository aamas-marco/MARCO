from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
# from envs.mb_switch_env_multi import SwitchWithModel
from envs.mb_switch_env_bridge import SwitchWithModel
import torch

device = torch.device('cuda')

class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        # assert self.batch_size > 1
        self.action_size = self.args.n_switches * 2 + 2
        self.env = None # env is initialized after model learnt with random policy at t=0

        if args.episode_limit > 0:
            self.episode_limit = args.episode_limit
        # default from learning2communication paper
        else:
            if args.n_bridges > 0:
                self.episode_limit = 4 * args.n_agents - 6 + args.n_bridges
            else:
                self.episode_limit = 4 * args.n_agents - 6

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.max_returns = []


        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()

        self.env.reset()

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }

        for i in range(self.batch_size):
            pre_transition_data["state"].append(self.env.get_state()[i]) #(43,)
            pre_transition_data["avail_actions"].append(self.env.get_avail_actions()[i]) #(4,8)
            pre_transition_data["obs"].append(self.env.get_obs()[i]) #(4,7)

        # sanity check with mf switch env
        # for i in range(self.batch_size):
        #     pre_transition_data["state"].append(self.env.get_state())
        #     pre_transition_data["avail_actions"].append(np.array(self.env.get_avail_actions()))
        #     pre_transition_data["obs"].append(np.array(self.env.get_obs()))

        try:
            self.batch.update(pre_transition_data, ts=0)
        except:
            import pdb; pdb.set_trace()
        self.t = 0
        self.env_steps_this_run = 0


    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated,
                                               test_mode=test_mode) #(1,4)

            # import pdb; pdb.set_trace()

            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }

            actions_padded = (torch.ones((self.batch_size, self.args.n_agents))*(self.action_size-1)).to(device) # to send to the enviornment
            for i, b in enumerate(envs_not_terminated):
                actions_padded[b] = actions[i]

            try:
                self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            except:
                import pdb; pdb.set_trace()
            # send actions to each env
            reward, done, env_info = self.env.step(actions_padded)  #(1,1), (1,1)

            #sanity check
            # reward, done, env_info = self.env.step(cpu_actions[0])
            # reward = [reward]
            # done = [done]
            # import pdb; pdb.set_trace()

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)

            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            s_t = self.env.get_state() #(1,43)
            o_t = self.env.get_obs() #(1,4,7)
            avail_t = self.env.get_avail_actions() #(1,4,8)

            # sanity check
            # s_t = np.array(self.env.get_state()).reshape(1,43)
            # o_t = np.array(self.env.get_obs()).reshape(1,4,7)
            # avail_t = np.array(self.env.get_avail_actions()).reshape(1,4,8)

            # Receive data back for each unterminated env
            for idx in range(self.batch_size):
                if not terminated[idx]:
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((reward[idx],))
                    episode_returns[idx] += reward[idx]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if done[idx]:
                        env_terminated = True
                    terminated[idx] = done[idx]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(s_t[idx])
                    pre_transition_data["avail_actions"].append(avail_t[idx])
                    pre_transition_data["obs"].append(o_t[idx])

            # Add post_transiton data into the batch
            try:
                self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            except:
                import pdb; pdb.set_trace()
            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            try:
                self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)
            except:
                import pdb; pdb.set_trace()
        if not test_mode:
            self.t_env += self.env_steps_this_run


        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        if self.env.exploration:
            log_prefix = "exp_" + log_prefix
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)


        cur_returns.extend(episode_returns)
        try:
            self.max_returns.extend(self.env.max_reward) # for normalization
        except:
            self.max_returns.append(self.env.max_reward)
        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size


        # if test_mode and (len(self.test_returns) == n_test_runs):
        #     self._log(cur_returns, cur_stats, log_prefix)
        if self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        assert(len(self.max_returns) == len(returns)) # sanity check
        return_normalized = (np.mean(returns) + 1) / (np.mean(self.max_returns) + 1 + 1e-8) # move [-1,1] to [0,2]
        self.logger.log_stat(prefix + "return_mean", return_normalized, self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)

        # clear
        returns.clear()
        self.max_returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()