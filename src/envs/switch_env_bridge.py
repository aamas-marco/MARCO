import numpy as np
import torch
from torch.nn import functional as F

import copy
import math

# generalizing to any number of agents and any number of switches
class SwitchGame:
    def __init__(self, n_agents=3, n_switches=1, n_bridges=3, include_actions=True, episode_limit=-1):
        '''
        Initializes the Switch Game with given parameters.
        '''

        # Set game defaults
        self.n_agents = n_agents
        self.n_switches = n_switches
        self.n_bridges = n_bridges

        if episode_limit > 0:
            self.episode_limit = episode_limit
        # default from learning2communication paper + bridge length
        else:
            self.episode_limit = 4 * self.n_agents - 6 + self.n_bridges

        self.reward_all_live = 1
        self.reward_all_die = -1

        # state size:
        self.state_size = self.n_switches + (self.n_agents + 1) + (self.n_agents) + (self.n_bridges + 1) * self.n_agents
        # action size per agent: (turn on switch, turn off switch) for each switch, tell, none, left, right, suicide
        self.action_size = 2 * self.n_switches + 2 + 3
        # obs size per agent
        self.obs_size = 1 + 2 * self.n_switches + (self.n_bridges + 1)
        self.reset()

        self.games = 0
        self.win = 0
        self.loose = 0
        self.draw = 0

        # for code readability
        self.tell_idx = self.n_switches * 2
        self.none_idx = self.n_switches * 2 + 1
        self.left_idx = self.n_switches * 2 + 2
        self.right_idx = self.n_switches * 2 + 3
        self.suicide_idx = self.n_switches * 2 + 4

        self.include_actions = include_actions
        self.max_reward = 1
        self.crossed_bridge = [False]*self.n_agents #which agent crossed the bridge
        self.start_time = -1 # timestep when all agents all crosses bridge

        self.mf = True  # for sanity check purposes

    def refresh_game_tally(self):
        self.games = 0
        self.win = 0
        self.loose = 0
        self.draw = 0

    def reset(self, active_agent=None):
        """
        Resets the environment for the next episode and sets up the agent sequence for the next episode.
        """
        self.cur_step = 0
        self.has_been = np.zeros((self.n_agents))

        self.bridge_position = np.zeros((self.n_agents))
        self.crossed_bridge = [False]*self.n_agents
        self.start_time = -1

        if active_agent is None: # for debugging purposes
            self.active_agent = np.zeros((self.episode_limit - self.n_bridges + 2)) # collects one more after episode terminated
            # determine which agent goes into room at each step
            for step in range(self.episode_limit - self.n_bridges + 2):
                agent_id = np.random.randint(self.n_agents)
                self.active_agent[step] = agent_id
        else:
            self.active_agent = active_agent

        self.last_action = np.zeros((self.n_agents, self.action_size)).flatten()
        self.max_reward = 1

        self.switch = [0] * self.n_switches

        # (n_actions + n_agents*2) n_actions: (switch on/off)*n_actions, n_actions + 1~3: in room agent ID, n_actions+4~6: if each agent has been in room
        # switch is off
        switch = np.array(self.switch)
        # no one is in room
        in_room = np.zeros((self.n_agents+1))
        in_room[-1] = 1
        # no one has been in the room
        has_been = np.zeros((self.n_agents))
        # everyone starts at begining of bridge
        pos_in_bridge = np.zeros((self.n_agents, self.n_bridges+1))
        pos_in_bridge[:, 0] = 1

        self.state = np.concatenate((switch, in_room, has_been, pos_in_bridge.flatten()), axis=-1)

        # (n_agents, 3) 0: in room/None, (1: switch on, 2: switch off) * n_actions, position in bridge
        self.obs = [np.concatenate((np.zeros(1+2*self.n_switches), pos_in_bridge[i])) for i in range(self.n_agents)]

        # see whether all agents can go into the room to determine max reward
        tmp = len(np.unique(np.array(self.active_agent[:self.episode_limit - self.n_bridges])).tolist()) # off by one error

        if tmp < self.n_agents:
            self.max_reward = 0
        else:
            self.max_reward = 1

        # self.state[int(self.n_switches + self.active_agent[0])] = 1
        # self.has_been[int(self.active_agent[0])] = 1
        #
        # # update obs: who is first to be in room, and who can see the switch is off
        # self.obs[int(self.active_agent[0])][0] = 1 # update in room
        # for i in range(self.n_switches):
        #     self.obs[int(self.active_agent[0])][1+2*i+1] = 1 # switch is off
        return self

    def get_obs(self):
        return [self.obs[i].flatten() for i in range(self.n_agents)]

    def get_state(self):
        if not self.include_actions:
            return self.state.astype(np.float32)
        state = self.state

        if torch.is_tensor(state):
            state = state.cpu().data.numpy().astype(np.float32)
        actions = self.last_action
        if torch.is_tensor(actions):
            actions = actions.cpu().data.numpy().astype(np.float32)

        state = np.concatenate((state, actions), axis=0)
        return state

    def get_obs_size(self):
        return self.obs_size

    def get_state_size(self):
        if not self.include_actions:
            return self.state_size
        return self.state_size + self.action_size * self.n_agents

    def get_avail_actions(self):
        '''
        :return:(n_agents,n_actions)
        '''
        # all agents can perform "None" action at all times
        avail = np.zeros((self.n_agents, self.action_size))
        avail[:, 2 * self.n_switches + 1] = 1

        # if all agents have crossed the bridge:
        if all(self.crossed_bridge):
            assert self.start_time >= 0
            active_agent_idx = int(self.active_agent[self.cur_step-self.start_time])
            avail[active_agent_idx]= np.ones((self.action_size))

        # if some agents are still on the bridge
        else:
            #avail actions when agents are on bridge: none (already taken care of), and left, right, suicide
            avail_action_idx_bridge = np.arange(3) + (self.n_switches*2 + 2)
            avail[:, avail_action_idx_bridge] = 1

        # check if no actions available (this should never happen though)
        for i in range(self.n_agents):
            if np.array(avail[i]).sum() == 0:
                raise ValueError("no actions available in switch mf env")

        avail = avail.astype(int)
        return [avail[i].tolist() for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        # (on, off)*n_switches, tell, none, left, right, suicide
        return self.action_size

    def get_reward(self, a_t_indices, a_t_idx, active_agent_idx):
        """
        Returns the reward for action a_t taken by current agent in state a_t
        """

        # automaticlaly dies
        if self.suicide_idx in a_t_indices:
            self.loose += 1
            self.games += 1
            return self.reward_all_die, True


        if a_t_idx == self.tell_idx:  # if the active agent TELL
            terminal = True
            # check if all agent has been in the room
            tmp = copy.deepcopy(self.has_been)
            tmp[active_agent_idx] = 1 # set this to 1 because has_been does not account for timestep t

            # all agents visited room
            if np.sum(tmp) == self.n_agents:
                reward = self.reward_all_live
                self.win += 1
                self.games += 1
            else:
                reward = self.reward_all_die
                self.loose += 1
                self.games += 1

        elif self.cur_step >= self.episode_limit-1:  # if it didn't tell, but episode ended
            terminal = True
            reward = 0
            self.draw += 1
            self.games += 1

        else:  # episode did not end
            terminal = False
            reward = 0


        return reward, terminal

    def step(self, a_t_indices):
        '''
        :param a_t_indices: (n_agents)
        :return:
        '''
        active_agent_idx = -1
        # if crossed bridge, update switch if necessary
        if all(self.crossed_bridge):
            assert self.start_time >= 0
            active_agent_idx = int(self.active_agent[self.cur_step-self.start_time])
            # only the active agent's action matters
            a_t_idx = a_t_indices[active_agent_idx]

            # update light switch. actions are (on, off)*n, tell, none
            if a_t_idx < self.tell_idx:
                chosen_switch = int(math.floor(a_t_idx/2.0))
                if a_t_idx % 2 == 0: # turn on chosen switch
                    self.switch[chosen_switch] = 1
                else: # turn off chosen switch
                    self.switch[chosen_switch] = 0
        # if still crossing bridge, update agents' position in bridge if necessary, and check if all have crossed
        # if all is crossed we should not be able to update bridge positions anymore
        else:
            # update position
            for agent_idx in range(self.n_agents):
                a_t_idx = a_t_indices[agent_idx]
                if a_t_idx == self.left_idx:
                    self.bridge_position[agent_idx] -= 1
                elif a_t_idx == self.right_idx:
                    self.bridge_position[agent_idx] += 1

            self.bridge_position = np.clip(self.bridge_position, 0, self.n_bridges)

            # check if all crossed:
            self.crossed_bridge = [self.bridge_position[i] == self.n_bridges for i in range(self.n_agents)]
            if all(self.crossed_bridge):
                self.start_time = self.cur_step + 1

        # get reward and term
        reward, terminal = self.get_reward(a_t_indices, a_t_idx, active_agent_idx)

        # update last_action in one hot form (n_agents, num_actions):
        self.last_action = F.one_hot(torch.tensor(a_t_indices, dtype=int).flatten(), num_classes=self.action_size).flatten().float()

        # update step counter
        self.cur_step += 1

        # Don't return yet, update the obs to put into the buffer


        position_in_bridge = F.one_hot(torch.tensor(self.bridge_position, dtype=int).flatten(), num_classes=self.n_bridges+1).flatten().float()
        switch = np.array(self.switch)
        if all(self.crossed_bridge):
            in_room = np.zeros((self.n_agents + 1))

            # update state obs for next timestep if its not the end
            active_agent_idx_prev = active_agent_idx
            active_agent_idx = int(self.active_agent[self.cur_step-self.start_time])
            # update state for next timestep
            in_room[active_agent_idx] = 1

            if active_agent_idx_prev >= 0:
                self.has_been[active_agent_idx_prev] = 1


        else:
            # no one is in room
            in_room = np.zeros((self.n_agents+1))
            in_room[-1] = 1

        self.state = np.concatenate((switch, in_room, copy.deepcopy(self.has_been), position_in_bridge), axis=0)
        assert self.state.shape[0] == self.state_size

        # update obs
        for i in range(self.n_agents):
            if i == active_agent_idx:
                tmp = np.zeros((self.n_switches, 2))
                tmp[np.arange(self.n_switches), 1-np.array(self.switch).astype(int)] = 1
                self.obs[i] = np.concatenate((np.array([1]), tmp.flatten(),
                                              position_in_bridge.reshape(self.n_agents, -1)[i]))
            else:
                self.obs[i] = np.concatenate((np.zeros((1+2*self.n_switches)),
                                              position_in_bridge.reshape(self.n_agents, -1)[i]))

        return reward, terminal, {} # leave info empty for now


    def render(self):
        return None

    def close(self):
        self.reset()
        return None

    def seed(self):
        return None

    def save_replay(self):
        return None

    def get_stats(self):
        stats = {}
        return stats

    def get_env_info(self):

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}

        return env_info