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
from learn_model_mpe import NetworkMPE
import random
from envs.mpe_env import make_env



# TODO: check inclusion for last action in real environment in state and obs

working_directory = '/home/MARCO'
assert os.path.isdir(working_directory) == True

device = torch.device('cuda')



class MpeWithModel():

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

        print ("********************prefix", prefix)
        # update models
        # state, action -> state'
        state_state_models = []
        for i in range(self.ensemble):
            state_state = NetworkMPE(self.state_size+self.action_size*self.n_agents, hidden_size=500)
            state_state.load_state_dict(
                copy.deepcopy(torch.load(prefix + '/state_state_' + str(i) + '.pt', map_location=device)))
            state_state_models.append(state_state.to(device))

        # # state_obs
        # state_obs = Network(self.state_size+self.action_size, self.n_agents*3, 100, "state_obs", self.reward_type, mpe=True)
        # state_obs.load_state_dict(copy.deepcopy(torch.load(prefix + '/state_obs.pt', map_location=device)))

        # state, last_state action -> reward
        state_reward_models = []
        for i in range(self.ensemble):
            state_reward = Network(self.state_size+self.state_size+self.action_size*self.n_agents, 1, 500, "state_reward", self.reward_type,
                                    mpe=True)
            state_reward.load_state_dict(torch.load(prefix + '/state_reward_' + str(i) + '.pt', map_location=device))
            state_reward_models.append(state_reward.to(device))


        # state -> done #
        # state_done = Network(self.state_size+self.action_size, 1, 100, "state_done", self.reward_type,  mpe=True)
        # state_done.load_state_dict(torch.load(prefix + '/state_done.pt', map_location=device))

        # # state -> available
        # state_avail = Network(self.state_size, self.n_agents*4, 100, "state_avail", self.reward_type, switch=True)
        # state_avail.load_state_dict(torch.load(prefix + '/state_avail.pt', map_location=device))

        self.state_state_models = state_state_models
        self.state_reward_models = state_reward_models

        return state_state_models, state_reward_models



    def __init__(self, env_path='models_all', reward_type="regression", ensemble=1, n_agents=2, exploration=False, beta1=0, beta3=0, buffer=None, bridge=False):
        # get env models of the latest timestep
        self.env_path = env_path
        self.reward_type = reward_type
        self.ensemble = ensemble

        # env attributes
        self.n_agents = n_agents

        self.episode_limit = 25

        if not bridge:
            self.state_size = 42 # the last self.n_agents * 4 are the last action, dont need this, only cur action
        else:
            self.state_size = 46 # the last self.n_agents * 4 are the last action, dont need this, only cur action

        self.action_size = 50
        self.obs_size = 21

        self.exploration = exploration
        self.beta1 = beta1
        self.beta3 = beta3
        self.baseline = 0 # aka beta 2

        self.true_env = make_env("simple_reference")

        # env models
        state_state_models, state_reward_models = self.refresh_env()
        self.state_state_models = state_state_models
        self.state_reward_models = state_reward_models
        # self.state_done = state_done

        # other args
        self.reward_type = reward_type
        self.ensemble = ensemble

        self.reset()

        self.games = 0

        self.mb = True


        # self.buffer = buffer
        # if self.buffer is not None:
        #     self.set_baseline()


    def get_r_tilde(self, x_state, x_act):
        '''
        :param x_state:
        :param x_act:
        :return: (normalized) variance (1)
        '''
        with torch.no_grad():
            x = torch.cat((x_state, x_act), axis=0)
            predictions = []
            predictions_state = []
            for i, model in enumerate(self.state_state_models):
                p = model.sample(x).flatten()
                x_r = torch.cat((x_state, p, x_act))
                r = self.state_reward_models[i](x_r).cpu().numpy()
                predictions.append(r)
                predictions_state.append(p.flatten().cpu().numpy())


            predictions = np.array(predictions)
            predictions_state = np.array(predictions_state)

            variances = np.var(predictions, axis=0)
            variances_state = np.var(predictions_state, axis=0)
            # if variances.shape[-1] != 1 + 2 * self.n_agents:
            #     raise ValueError("shape of variances: ", variances.shape)

            # todo: try others than max
            # r_tilde = np.max(variances)
            # r_tilde = max(0, r_tilde-self.baseline)
            r_tilde = np.mean(variances / np.abs(np.mean(predictions))) + np.mean(variances_state)

        return r_tilde


    # def set_buffer(self, buffer):
    #     self.buffer = buffer

    def set_baseline(self, baseline):
        self.baseline = baseline

    # def set_baseline(self):
    #     if self.ensemble <= 1:
    #         self.baseline = 0
    #         return 0
    #     else:
    #         buffer_baseline = []
    #         for i in range(self.buffer['state'].shape[0]):
    #             for j in range(self.buffer['state'][i].shape[0]):
    #                 if self.buffer['filled'][i][j] == 0:
    #                     pass
    #                 else:
    #                     actions = self.buffer['actions'][i][j]
    #                     x_act = F.one_hot(torch.tensor(actions, dtype=int).flatten(), num_classes=4).flatten().float()
    #                     x_state = self.buffer['state'][i][j][:self.n_agents * 2 + 1]
    #                     x = torch.cat((x_state, x_act), axis=0)
    #                     predictions = []
    #                     for model in self.state_state_models:
    #                         p = model.sample(x).cpu().data.numpy().flatten()
    #                         predictions.append(p)
    #                     predictions = np.array(predictions)
    #                     variances = np.var(predictions, axis=0)
    #                     buffer_baseline.append(np.max(variances))
    #         self.baseline = np.mean(np.array(buffer_baseline))
    #         return self.baseline

    def refresh_game_tally(self):
        self.games = 0

    def reset(self):
        """
        Resets the environment for the next episode and sets up the agent sequence for the next episode.
        """

        self.cur_step = 0
        self.last_state = np.zeros(self.state_size)
        self.last_action = np.zeros((self.n_agents, 50)).flatten()
        # self.state = np.zeros((self.state_size))
        # self.obs = [np.zeros(21) for _ in range(self.n_agents)]


        self.obs = self.true_env.reset()
        self.state = self.true_env.get_state()

        return self

    def step(self, a_t_indices):

        info = {}  # hard code
        x_a_t = F.one_hot(torch.tensor(a_t_indices, dtype=int).flatten(), num_classes=50).flatten().float().cuda()
        x_s_t = torch.tensor(self.state.flatten()).float().cuda()

        # TODO: done is wrong
        with torch.no_grad():

            # get next state x_s_t
            m = np.random.randint(self.ensemble)
            state_state = self.state_state_models[m]
            x = torch.cat((x_s_t, x_a_t), axis=0)
            s_tplus1 = state_state.sample(x).cpu().data.numpy().flatten()
            self.state = s_tplus1.reshape(self.state_size,)

            # get reward (for t+1)
            x_s_tplus1 = torch.tensor(s_tplus1.flatten()).float().cuda()
            x = torch.cat((x_s_t, x_s_tplus1, x_a_t), axis=0)
            if True:
                output = self.state_reward_models[m](x).cpu().data.numpy()
                reward_env = output
            else:
                a1_goal = torch.argmax(x_s_tplus1[29:32])
                a2_goal = torch.argmax(x_s_tplus1[8:11])
                a1_vals = x_s_tplus1[(a1_goal+1)*2:(a1_goal+2)*2]
                a2_vals = x_s_tplus1[(a2_goal+1)*2+21:(a2_goal+2)*2+21]
                output = torch.square(a1_vals[0]) + torch.square(a1_vals[1]) + torch.square(a2_vals[0]) + torch.square(a2_vals[1])
                reward_env = -output


            # get done (for t+1)
            # x = torch.cat((x_s_tplus1, x_a_t), axis=0) #right
            # # x = torch.cat((x_s_t, x_a_t), axis=0)
            # output = self.state_done(x).cpu().data.numpy()
            # y_pred = (output >= 0.5).astype(int)
            # done = bool(y_pred)
            done = False

            if self.cur_step >= self.episode_limit - 1:
                done = True

            #update observation (for t+1)
            self.obs = [self.state[:21], self.state[21:42]]
            # self.obs = self.state.split(21)
            # print ("cur obs: ", self.obs)

        self.cur_step += 1

        if done:
            self.games = self.games + 1
            self.reset()

        self.last_action = x_a_t.cpu().data.numpy().flatten()
        self.last_state = x_s_t.cpu().data.numpy().flatten()

        if self.exploration:
            if self.beta3 > 0:
                assert self.ensemble > 1
                r_tilde = self.get_r_tilde(x_s_t, x_a_t)
                # lmbda = min(1, self.beta3 * r_tilde)
                # r_explore = (1 - lmbda) * reward_env + (2 * lmbda - 1) * r_tilde
                reward = reward_env + self.beta3 * r_tilde
                info = {"reward_env_exp": copy.deepcopy(reward_env), "reward_tilde_exp": copy.deepcopy(r_tilde)}
            else:
                reward = reward_env
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
        return [self.obs[i].flatten() for i in range(2)]


    def get_obs_agent(self, agent_id):
        return None


    def get_obs_size(self):
        return 3


    def get_state(self):
        return self.state.astype(np.float32)


    def get_state_size(self):
        return self.state_size


    def get_avail_actions(self): # get available agent actions
        # x_state = torch.tensor(self.state.flatten()).float()
        # x_state = x_state[:self.state_size]
        # output = self.state_avail(x_state).cpu().data.numpy()
        # y_pred = (output >= 0.5).astype(int)
        # actions = y_pred.reshape(self.n_agents, 4).tolist()
        # for i in range(self.n_agents):
        #     if np.array(actions[i]).sum() == 0:
        #         # todo: make all actions available
        #         print ("**************NO AVAILABLE ACTIONS***********")
        #         self.relearn = True
        #         actions[i] = [0, 0, 0, 1]
        # return actions
        return [[1]*50]*2


    def get_avail_agent_actions(self, agent_id):
        return None


    def get_total_actions(self):
        return 50


    def render(self):
        return None


    def close(self):
        self.reset()
        self.games = 0
        return None


    def seed(self):
        return None


    def save_replay(self):
        return None

    def get_stats(self):
        return {}


    def get_env_info(self):

        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": 21,
                    "n_actions": 50,
                    "n_agents": 2,
                    "episode_limit": self.episode_limit}

        return env_info