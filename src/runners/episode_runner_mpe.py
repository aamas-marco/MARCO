from runners.episode_runner import EpisodeRunner

from envs.mpe_env import make_env

import numpy as np

class DataCollector(EpisodeRunner):

    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.env = make_env("simple_reference")

    def run(self, total_timesteps, test_mode=True, epsilon=0.1, all_model=False):
        '''
        collect data
        '''
        data = []
        i = 0
        while True:
            self.reset()
            terminated = False
            self.mac.init_hidden(batch_size=self.batch_size)
            while not terminated:
                i += 1
                pre_transition_data = {
                    "state": [self.env.get_state()],
                    "avail_actions": [self.env.get_avail_actions()],
                    "obs": [self.env.get_obs()]
                }
                s_t = np.copy(pre_transition_data["state"]).flatten()
                avail_t = np.copy(pre_transition_data["avail_actions"]).flatten()
                o_t = np.copy(pre_transition_data["obs"]).flatten()
                a_tminus1 = self.env.last_action.flatten()

                self.batch.update(pre_transition_data, ts=self.t)

                # Pass the entire batch of experiences up till now to the agents
                # Receive the actions for each agent at this timestep in a batch of size 1
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=True, epsilon=epsilon)
                a_t = np.copy(actions[0].data.cpu().numpy()).flatten()
                reward, terminated, env_info = self.env.step(actions[0])

                post_transition_data = {
                    "actions": actions,
                    "reward": [(reward,)],
                    "terminated": [(terminated != env_info.get("episode_limit", False),)],
                }
                r_t = np.copy(post_transition_data["reward"]).flatten()
                t = np.array([self.t])
                s_tplus1 = self.env.get_state().flatten()
                term = np.copy([terminated]).flatten()
                avail_tplus1 = np.copy(self.env.get_avail_actions()).flatten()
                o_tplus1 = np.copy(self.env.get_obs()).flatten()

                self.batch.update(post_transition_data, ts=self.t)

                self.t += 1

                if all_model:
                    data_point = np.concatenate((s_t, s_tplus1, a_t, r_t, avail_tplus1, o_tplus1, term, t, a_tminus1),
                                                axis=0)
                else:
                    data_point = np.concatenate((s_t, s_tplus1, a_t, r_t, avail_t, o_t, term, t, a_tminus1), axis=0)
                data.append(data_point)

            last_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]

            }
            self.batch.update(last_data, ts=self.t)

            # Select actions in the last stored state
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            self.batch.update({"actions": actions}, ts=self.t)

            if i >= total_timesteps:
                return np.array(data)