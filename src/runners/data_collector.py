from runners.episode_runner import EpisodeRunner

from envs.switch_env_multi import SwitchGame
from envs.switch_env_bridge import SwitchGame as SwitchGameWithBridge
import numpy as np
import torch
import torch.nn.functional as F



class DataCollector(EpisodeRunner):

    def __init__(self, args, logger):
        super().__init__(args, logger)
        if args.task == "smac":
            pass
        elif args.task == "switch":
            if args.n_bridges > 0:
                self.env = SwitchGameWithBridge(n_agents=args.n_agents, n_switches=args.n_switches, n_bridges=args.n_bridges,
                                      episode_limit=args.episode_limit)
            else:
                self.env = SwitchGame(n_agents=args.n_agents, n_switches=args.n_switches, episode_limit=args.episode_limit)
        else:
            raise ValueError("task not specified")

    def run(self, total_timesteps, test_mode=True, epsilon=0.1, all_model=False, frame_stacking=False, rnn_state_state=False):
        '''
        collect data
        '''
        data = []
        data_rnn = []
        i = 0
        ep = 0
        while True:
            self.reset()
            terminated = False
            self.mac.init_hidden(batch_size=self.batch_size)

            if frame_stacking:
                last4actions = np.zeros((4, 27))
                last4states = np.zeros((4, 21))
            if rnn_state_state:
                dp_x = []
                dp_y = []
                dp_term = []
                dp_mask = []
            # print ("ep: ", ep)
            ep+=1
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
                    if frame_stacking:
                        data_point = np.concatenate((s_t, s_tplus1, a_t, r_t, avail_t, o_t, term, t, a_tminus1,
                                                     last4actions.flatten(), last4states.flatten()), axis=0)
                    else:
                        data_point = np.concatenate((s_t, s_tplus1, a_t, r_t, avail_t, o_t, term, t, a_tminus1), axis=0)
                        # import pdb; pdb.set_trace()

                if rnn_state_state:
                    x_a_t = F.one_hot(torch.tensor(a_t, dtype=int).flatten(),
                                      num_classes=9).flatten().float().cpu().data.numpy()
                    dp_x.append(np.concatenate((s_t[:21], x_a_t), axis=0))
                    dp_y.append(s_tplus1[:21])
                    dp_term.append(term)

                data.append(data_point)

                if frame_stacking:
                    x_a_t = F.one_hot(torch.tensor(a_t, dtype=int).flatten(),
                                      num_classes=9).flatten().float().cpu().data.numpy()

                    last4actions = last4actions[1:]
                    last4actions = np.concatenate((last4actions, x_a_t.reshape(1, -1)), axis=0)

                    last4states = last4states[1:]
                    last4states = np.concatenate((last4states, s_t[:21].reshape(1, -1)), axis=0)

            last_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]

            }
            # import pdb; pdb.set_trace()
            if rnn_state_state:
                assert len(dp_x) == self.t
                dp_mask = np.zeros(60)
                dp_mask[np.arange(self.t)] = 1
                padding = 60 - self.t

                dp_x = np.array(dp_x)
                dp_y = np.array(dp_y)
                dp_term = np.array(dp_term)

                dp_x = np.concatenate((dp_x.flatten(), np.zeros(padding*(48))), axis=0)
                dp_y = np.concatenate((dp_y.flatten(), np.zeros(padding*(21))), axis=0)
                dp_term = np.concatenate((dp_term.flatten(), np.zeros(padding)), axis=0)

                data_rnn.append(np.concatenate((dp_x, dp_y, dp_term, dp_mask), axis=0))

            # import pdb; pdb.set_trace()

            self.batch.update(last_data, ts=self.t)
            # Select actions in the last stored state
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            self.batch.update({"actions": actions}, ts=self.t)
            self.t_env += self.t

            if i >= total_timesteps:
                if rnn_state_state:
                    return np.array(data), np.array(data_rnn)
                return np.array(data), None
