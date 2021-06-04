import sys
import numpy as np
import os
import copy

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from learn_model import Network
from learn_model_switch import DensityNetworkSwitch_state
import random



# TODO: check inclusion for last action in real environment in state and obs

working_directory = '/home/MARCO'
assert os.path.isdir(working_directory) == True

device = torch.device('cuda')



class SwitchWithModel():

    # get newest trained model, there should be at least model for t_env=0
    def refresh_env(self):
        '''
        :param env_path: path of env models of each timestep
        :param prefix: working directory
        :return: env models
        '''

        prefix = os.path.join(working_directory, self.env_path)
        # get lastest model path
        timesteps = []
        for name in os.listdir(prefix):
            # Check if they are dirs the names of which are numbers
            if name.isdigit():
                timesteps.append(int(name))
        if len(timesteps) == 0:
            raise ValueError("where are the initial models??")

        last_timestep = max(timesteps)
        prefix = os.path.join(prefix, str(last_timestep))
        print ("fetching model based enviornment using", prefix)

        possible_rewards = np.array([-0, 1, -1])

        print ("********************prefix", prefix)
        # update models
        # state, action -> state'
        state_state_models = []
        for i in range(self.ensemble):
            state_state = DensityNetworkSwitch_state(self.state_size + self.action_size, hidden_size=500, n_agents=self.n_agents,
                                                     n_switches=self.n_switches, n_bridges=0)
            state_state.load_state_dict(
                copy.deepcopy(torch.load(prefix + '/state_state_' + str(i) + '.pt', map_location=device)))
            state_state.to(device)
            state_state_models.append(state_state)

        # state_obs
        state_obs = Network(self.state_size+self.action_size, self.obs_size, 500, "state_obs", self.reward_type, switch=True)
        state_obs.load_state_dict(copy.deepcopy(torch.load(prefix + '/state_obs.pt', map_location=device)))
        state_obs.to(device)

        # state, last_state action -> reward
        if self.reward_type == "category":
            state_reward = Network(self.state_size+self.state_size+self.action_size, len(possible_rewards), 500, "state_reward",
                                   self.reward_type, switch=True)
            state_reward.load_state_dict(torch.load(prefix + '/state_reward_category.pt', map_location=device))
        else:
            state_reward = Network(self.state_size+self.state_size+self.action_size, 1, 500, "state_reward", self.reward_type,
                                   switch=True)
            state_reward.load_state_dict(torch.load(prefix + '/state_reward_regression.pt', map_location=device))
        state_reward.to(device)

        # state -> done #
        state_done = Network(self.state_size+self.action_size, 1, 500, "state_done", self.reward_type,  switch=True)
        state_done.load_state_dict(torch.load(prefix + '/state_done.pt', map_location=device))
        state_done.to(device)

        # state -> available
        state_avail = Network(self.state_size, self.action_size, 500, "state_avail", self.reward_type, switch=True)
        state_avail.load_state_dict(torch.load(prefix + '/state_avail.pt', map_location=device))
        state_avail.to(device)

        return state_state_models, state_obs, state_reward, state_done, state_avail, possible_rewards



    # TODO: implement ensembles
    def __init__(self, env_path='models_all', reward_type="category", ensemble=1, n_agents=3, n_switches=1, batch_size=1,
                 exploration=False, beta1=0, beta3=0, buffer=None, episode_limit=0):
        # get env models of the latest timestep
        self.env_path = env_path
        self.reward_type = reward_type
        self.ensemble = ensemble
        self.batch_size = batch_size

        # env attributes
        self.n_agents = n_agents
        self.n_switches = n_switches

        if episode_limit > 0:
            self.episode_limit = episode_limit
        else:
            self.episode_limit = 4 * self.n_agents - 6 # default from learning2communication paper
        self.reward_all_live = 1
        self.reward_all_die = -1


        self.state_size = self.n_switches + self.n_agents * 2  # the last self.n_agents * 4 are the last action, dont need this, only cur action
        self.action_size = (2 * self.n_switches + 2) * self.n_agents
        self.obs_size = (1 + 2 * self.n_switches) * self.n_agents


        self.exploration = exploration
        self.beta1 = beta1
        self.beta3 = beta3
        self.baseline = 0 # aka beta 2

        # env models
        state_state_models, state_obs, state_reward, state_done, state_avail, reward_mapping = self.refresh_env()
        self.state_state_models = state_state_models
        self.state_obs = state_obs
        self.state_reward = state_reward
        self.state_done = state_done
        self.state_avail = state_avail
        self.reward_mapping = np.array([0, 1, -1])

        # other args
        self.reward_type = reward_type
        self.ensemble = ensemble

        self.reset()

        self.mb = True
        self.include_actions = True

        self.buffer = buffer
        if self.buffer is not None:
            self.set_baseline()


    # TODO: batchify r_tilde
    def get_r_tilde(self, x_state, x_act):
        '''
        :param x_state:
        :param x_act:
        :return: (normalized) variance (1)
        '''
        return 0 
        # with torch.no_grad():
        #     x = torch.cat((x_state, x_act), axis=-1)
        #     predictions = []
        #     for model in self.state_state_models:
        #         p = model.sample(x).cpu().data.numpy().reshape(self.batch_size,-1)
        #         predictions.append(p)
        #
        #     predictions = np.array(predictions)
        #     variances = np.var(predictions, axis=-1)
        #     if variances.shape[-1] != self.state_size:
        #         raise ValueError("shape of variances: ", variances.shape)
        #
        #     # todo: try others than max
        #     r_tilde = np.max(variances, axis=-1)
        #     r_tilde = max(0, r_tilde-self.baseline)
        # return r_tilde


    def set_buffer(self, buffer):
        self.buffer = buffer

    # TODO: batchify
    def set_baseline(self, baseline):
        self.baseline = baseline

    def set_baseline(self):
        if self.ensemble <= 1:
            self.baseline = 0
            return 0
        else:
            buffer_baseline = []
            for i in range(self.buffer['state'].shape[0]):
                for j in range(self.buffer['state'][i].shape[0]):
                    if self.buffer['filled'][i][j] == 0:
                        pass
                    else:
                        actions = self.buffer['actions'][i][j]
                        x_act = F.one_hot(torch.tensor(actions, dtype=int).flatten(), num_classes=4).flatten().float()
                        x_state = self.buffer['state'][i][j][:self.n_agents * 2 + 1]
                        x = torch.cat((x_state, x_act), axis=0)
                        predictions = []
                        for model in self.state_state_models:
                            p = model.sample(x).cpu().data.numpy().flatten()
                            predictions.append(p)
                        predictions = np.array(predictions)
                        variances = np.var(predictions, axis=0)
                        buffer_baseline.append(np.max(variances))
            self.baseline = np.mean(np.array(buffer_baseline))
            return self.baseline

    def refresh_game_tally(self):
        self.games = 0
        self.win = 0
        self.loose = 0
        self.draw = 0

    def set_max_reward(self, active_agent_idx):
        # take into account of latest active agent, because has_been is not yet updated (?)
        tmp = copy.deepcopy(self.has_been)
        tmp[np.arange(self.batch_size), active_agent_idx] = 1
        tmp_max_reward = (np.sum(tmp, axis=-1) == self.n_agents).astype(int)

        # print ("check1", tmp_max_reward.shape, tmp.shape, self.max_reward.shape)

        # only set max reward for those that haven't yet terminated

        self.max_reward = (self.max_reward + (1 - self.done.astype(int)).flatten() * tmp_max_reward)

        # print ("check2, ", self.max_reward)
        # print("self done", self.done, "max reward", self.max_reward)
        self.max_reward = np.clip(self.max_reward, 0, 1)



    def reset(self):
        """
        Resets the environment for the next episode and sets up the agent sequence for the next episode.
        """

        self.max_reward = np.zeros(self.batch_size)
        self.cur_step = 0
        self.active_agent = np.zeros((self.batch_size, self.episode_limit + 1))
        self.last_state = np.zeros((self.batch_size, self.state_size))
        self.last_action = np.zeros((self.batch_size, self.n_agents*(2 * self.n_switches + 2)))
        self.has_been = np.zeros((self.batch_size, self.n_agents))
        self.done = np.zeros((self.batch_size, 1)).astype(bool)
        self.switch = np.zeros((self.batch_size, self.n_switches))

        # 0: switch on/off, 1-3: in room agent ID, 4-6: if each agent has been in room
        self.state = np.zeros((self.batch_size, self.state_size))

        # 0: in room/None, 1: switch on, 2: switch off
        self.obs = np.zeros((self.batch_size, self.n_agents, 1 + 2 * self.n_switches))

        # determine which agent goes into room at first step for initialization
        active = np.random.randint(0, self.n_agents, size=(self.batch_size)).astype(int)
        self.has_been[np.arange(self.batch_size), active] = 1

        # update state: switch is off, who is the first in room, the first one to have been in room
        self.state[np.arange(self.batch_size), active + self.n_switches] = 1

        # print ("*********obs: ", int(active), self.obs)
        # update obs: who is first to be in room, and who can see the switch is off
        self.obs[np.arange(self.batch_size), active, 0] = 1 # update in room

        for i in range(self.n_switches):
            self.obs[np.arange(self.batch_size), active, 1+2*i+1] = 1 # switch is off

        return self



    def step(self, a_t_indices):

        info = {}  # hard code
        x_a_t = F.one_hot(torch.tensor(a_t_indices, dtype=int).flatten(), num_classes=int(self.n_switches*2+2)).float().reshape(self.batch_size, -1)
        x_s_t = torch.tensor(self.state).float().reshape(self.batch_size, -1)

        # calculate max_reward
        try:
            active_onehot = self.state[:, self.n_switches:self.n_agents + self.n_switches].cpu().data.numpy().flatten()
        except:
            active_onehot = self.state[:, self.n_switches:self.n_agents + self.n_switches]

        assert np.all(np.sum(active_onehot, axis=-1), -1)  # check one agent in room at all time
        active = np.argmax(active_onehot, axis=-1).astype(int)
        # print(active, active_onehot)
        self.has_been[np.arange(self.batch_size), active] = 1

        # TODO: done is wrong
        with torch.no_grad():
            # get next state x_s_t
            chosen_model = np.random.randint(0, self.ensemble)
            state_state = self.state_state_models[chosen_model]
            x = torch.cat((x_s_t, x_a_t), axis=-1).to(device)
            s_tplus1 = state_state.sample(x).cpu().data.numpy().reshape(self.batch_size, self.state_size)
            self.state = s_tplus1

            # get reward (for t+1)
            x_s_tplus1 = torch.tensor(s_tplus1).float()
            x = torch.cat((x_s_t, x_s_tplus1, x_a_t), axis=-1).to(device)
            output = self.state_reward(x).cpu().data.numpy()
            if self.reward_type == "category":
                y_pred = np.argmax(output, axis=-1)
                reward_env = self.reward_mapping[y_pred].reshape(-1,1)
            else:
                reward_env = output

            # get done (for t+1)
            x = torch.cat((x_s_tplus1, x_a_t), axis=-1).to(device) #right
            # x = torch.cat((x_s_t, x_a_t), axis=0)
            output = self.state_done(x).cpu().data.numpy()
            done = (output >= 0.5).reshape(self.batch_size, -1)

            if self.cur_step >= self.episode_limit - 1:
                done = np.ones((self.batch_size,1)).astype(bool)

            # print ("check done: ", done, self.done)
            done = np.logical_or(done, self.done) # if we were done last timestep then we are also done now


            #update observation (for t+1)
            x = torch.cat((x_s_tplus1, x_a_t), axis=-1).to(device) # get the NEW observation
            output = self.state_obs(x).cpu().data.numpy()
            self.obs = (output >= 0.5).reshape(self.batch_size, self.n_agents, -1).astype(int)

            # update has been and max rewards
            active_agent_idx = np.argmax(x_s_t[:, self.n_switches:self.n_agents+self.n_switches].cpu().data.numpy(), axis=-1) # this step prob not necessary ?
            # import pdb; pdb.set_trace()
            self.set_max_reward(active_agent_idx)
            self.done = copy.deepcopy(done) # update done status after resetting max reward

            # TODO: check this

        self.cur_step += 1

        self.last_action = x_a_t.cpu().data.numpy().reshape(self.batch_size, -1)
        self.last_state = x_s_t.cpu().data.numpy().reshape(self.batch_size, -1)

        #TODO: convert to batched
        if self.exploration:
            assert self.ensemble > 1
            r_tilde = self.get_r_tilde(x_s_t, x_a_t)
            lmbda = min(1, self.beta3 * r_tilde)
            r_explore = (1 - lmbda) * reward_env + (2 * lmbda - 1) * r_tilde
            reward = r_explore
            info = {"reward_env_exp": copy.deepcopy(reward_env), "reward_tilde_exp": copy.deepcopy(r_tilde)}

        else:
            if self.ensemble > 1:

                r_tild = self.get_r_tilde(x_s_t, x_a_t)
                reward_task = reward_env - self.beta1 * r_tild
                reward = reward_task
                info = {"reward_env": copy.deepcopy(reward_env), "reward_tilde": copy.deepcopy(r_tild)}

            else:
                reward = reward_env

        return reward, done, info



    def get_obs(self):
        return self.obs.reshape(self.batch_size, self.n_agents, -1)


    def get_obs_agent(self, agent_id):
        return None


    def get_obs_size(self):
        return 1 + 2 * self.n_switches


    def get_state(self):
        if not self.include_actions:
            return self.state.astype(np.float32)

        state = self.state
        if torch.is_tensor(state):
            state = state.cpu().data.numpy().astype(np.float32)

        actions = self.last_action
        if torch.is_tensor(actions):
            actions = actions.cpu().data.numpy().astype(np.float32)

        state = np.concatenate((state, actions), axis=-1)
        return state


    def get_state_size(self):
        if not self.include_actions:
            return self.state_size
        return self.state_size + self.action_size


    def get_avail_actions(self): # get available agent actions
        x_state = torch.tensor(self.state).float().reshape(self.batch_size, self.state_size).to(device)

        output = self.state_avail(x_state).cpu().data.numpy()
        y_pred = (output >= 0.5).astype(int)
        actions = y_pred.reshape(self.batch_size, self.n_agents, -1).tolist()
        for j in range(self.batch_size):
            for i in range(self.n_agents):
                if np.array(actions[j][i]).sum() == 0:
                    # import pdb; pdb.set_trace()
                    print ("**************NO AVAILABLE ACTIONS***********")
                    self.relearn = True
                    tmp = np.zeros(actions.shape[-1])
                    tmp[-1] = 1
                    actions[j][i] = tmp
        # print (actions)
        return actions


    def get_avail_agent_actions(self, agent_id):
        return None


    def get_total_actions(self):
        return 2 * self.n_switches + 2



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
        stats = {
        }
        return stats


    def get_env_info(self):

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": 1 + 2 * self.n_switches,
                    "n_actions": 2 * self.n_switches + 2,
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}

        return env_info