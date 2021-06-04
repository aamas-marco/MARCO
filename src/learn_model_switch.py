import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Bernoulli, OneHotCategorical

import numpy as np
import pickle
import random
import sys
import os

from learn_model import get_new_dataset, Network, get_criterion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.multiprocessing as mp

working_directory = '/home/MARCO'
assert os.path.isdir(working_directory) == True



def preprocess(dataset, model_type='state_state', possible_rewards=[], reward_type='category', n_agents=3, n_switches=1,
               n_bridges=3, episode_limit=0):
    possible_rewards = [0, 1, -1] # overwrite always because rewards are known
    # take out the action information in the state

    if episode_limit > 0:
        episode_limit = episode_limit
    else:
        if n_bridges > 0:
            episode_limit = 4 * n_agents - 6 + n_bridges
        else:
            episode_limit = 4 * n_agents - 6


    if n_bridges > 0:
        obs_size = n_agents * (1 + 2 * n_switches + n_bridges + 1)
        num_actions = 2 * n_switches + 2 + 3
        action_size = n_agents * (num_actions)
        state_size = action_size + n_switches + (n_agents + 1) + (n_agents) + (n_bridges+1)*n_agents
    else:
        obs_size = n_agents * (1 + 2 * n_switches)
        num_actions = 2 * n_switches + 2
        action_size = n_agents * (num_actions)
        state_size = action_size + n_switches + n_agents * 2

    s_t = dataset[:, 0:state_size - action_size]
    s_tplus1 = dataset[:, state_size:state_size + state_size - action_size]
    a_t = dataset[:, state_size + state_size:state_size + state_size + n_agents]
    r_t = dataset[:, state_size + state_size + n_agents: state_size + state_size + n_agents + 1]
    avail_t = dataset[:, state_size + state_size + n_agents + 1:state_size + state_size + n_agents + 1 + action_size]
    o_t = dataset[:, state_size + state_size + n_agents + 1 + action_size: state_size + state_size + n_agents + 1 + action_size + obs_size]
    term = dataset[:, state_size + state_size + n_agents + 1 + action_size + obs_size:state_size + state_size + n_agents + 1 + action_size + obs_size + 1]
    t = dataset[:, state_size + state_size + n_agents + 1 + action_size + obs_size + 1:state_size + state_size + n_agents + 1 + action_size + obs_size + 1 + 1]
    a_tminus1 = dataset[:, state_size + state_size + n_agents + 1 + action_size + obs_size + 1 + 1:state_size + state_size + n_agents + 1 + action_size + obs_size + 1 + 1 + action_size]

    # s_t -> s_t+1
    if model_type == 'just_state':
        x = torch.tensor(s_t).float().to(device)
        y = torch.torch.tensor(s_tplus1).float().to(device)

    # s_t , a_t -> s_t+1
    elif model_type == 'state_state':
        x_s_t = torch.tensor(s_t).float().to(device)
        x_a_t = F.one_hot(torch.tensor(a_t, dtype=int).flatten(), num_classes=num_actions).float().to(device)
        x_a_t = x_a_t.reshape(-1, action_size)
        y = torch.torch.tensor(s_tplus1).float().to(device)
        x = torch.cat((x_s_t, x_a_t), axis=1)

    # s_t, a_t-1 -> o_t
    elif model_type == 'state_obs':
        x_s_t = torch.tensor(s_t).float().to(device)
        x_a_tminus1 = torch.tensor(a_tminus1).float().to(device)
        x = torch.cat((x_s_t, x_a_tminus1), axis=1)
        y = torch.torch.tensor(o_t).float().to(device)

    # s_t, s_t-1, a_t, r_t
    elif model_type == 'state_reward':

        x_s_t = torch.tensor(s_t).float().to(device)
        x_s_tplus1 = torch.tensor(s_tplus1).float().to(device)
        x_a_t = F.one_hot(torch.tensor(a_t, dtype=int).flatten(), num_classes=num_actions).float().to(device)
        x_a_t = x_a_t.reshape(-1, action_size)
        # print (x_state_last.shape, x_state.shape, x_act.shape)
        x = torch.cat((x_s_t, x_s_tplus1, x_a_t), axis=1)

        if reward_type == 'category':
            tmp = torch.torch.tensor(r_t).float().to(device).flatten()
            tmp[tmp == -1] = 2
            tmp = tmp.reshape(-1, 1)
            y = tmp.flatten().long().to(device)

        elif reward_type == 'regression':
            y = torch.torch.tensor(r_t).float().to(device).reshape(-1, 1)

        else:
            raise NameError("invalid reward type")

    # s_t -> terminated or not
    elif model_type == 'state_done':
        non_end_indices = np.where(t.flatten() != episode_limit-1)[0]
        x = np.take(s_tplus1, non_end_indices, 0) # TODO
        x_a_t = np.take(a_t, non_end_indices, 0)
        y = np.take(term, non_end_indices, 0)

        x_a_t = F.one_hot(torch.tensor(x_a_t, dtype=int).flatten(), num_classes=num_actions).float().to(device)
        x_a_t = x_a_t.reshape(-1, action_size)

        # print ("after take: **************", x.shape, x_a_t.shape)
        x = torch.cat((torch.tensor(x).to(device).float(), x_a_t.float()), axis=1)
        x = torch.tensor(x).float().to(device)
        y = torch.torch.tensor(y).float().to(device).reshape(-1, 1)

    # s_t -> available states
    elif model_type == 'state_avail':
        x = torch.tensor(s_t).float().to(device)
        y = torch.torch.tensor(avail_t).float().to(device)

    else:
        raise NameError('check model type')

    return x, y


class DensityNetworkSwitch_state(nn.Module):
    '''
    state to state
    '''
    def __init__(self, input_size, hidden_size=1000, n_agents=3, n_switches=1, n_bridges=3):
        super().__init__()
        self.n_agents = n_agents
        self.n_switches = n_switches
        self.n_bridges = n_bridges
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

        if n_bridges > 0:
            self.state_size = n_switches + (n_agents + 1) + (n_agents) + (n_bridges+1)*n_agents # does not include last action
            self.action_size = (2 * n_switches + 2 + 3) * self.n_agents
            ir_output_size = n_agents+1
        else:
            self.state_size = n_switches + n_agents * 2
            self.action_size = n_agents * (2 * n_switches + 2)
            ir_output_size = n_agents

        # predicts who is currently in room
        self.inroomNet = CategoricalNetworks(input_size, hidden_size=hidden_size, n_bridges=n_bridges, out_dim=ir_output_size,
                                             n_agents=self.n_agents, n_switches=self.n_switches).to(device)

        # predicts switch state, who has been in the room
        self.stateNet = StateNetwork(input_size, hidden_size=hidden_size, n_bridges=n_bridges, output_size=self.state_size-(ir_output_size),
                                         n_agents=self.n_agents, n_switches=self.n_switches).to(device)

    def forward(self, x):
        x = x.reshape(-1, self.state_size + self.action_size)
        inroom = self.inroomNet(x)
        state = self.stateNet(x)
        return inroom, state

    def loss(self, x, y):
        IR_dist, state = self.forward(x)

        y = y.reshape(-1, self.state_size)

        # in room
        if self.n_bridges > 0:
            y_IR = y[:, self.n_switches: self.n_switches+self.n_agents+1]
            # switch, has been, bridge positions
            y_state = torch.cat((y[:, 0:self.n_switches], y[:, self.n_switches+self.n_agents + 1:]), axis=1)
        else:
            y_IR = y[:, self.n_switches: self.n_switches+self.n_agents]
            # switch, has been, bridge positions
            y_state = torch.cat((y[:, 0:self.n_switches], y[:, self.n_switches+self.n_agents:]), axis=1)

        loss = -IR_dist.log_prob(y_IR) + self.criterion(state, y_state)
        return loss

    def sample(self, x):
        IRdist, state = self.forward(x)
        # who is in room
        if self.n_bridges > 0:
            IRsamples = IRdist.sample().reshape(-1, self.n_agents+1)
        else:
            IRsamples = IRdist.sample().reshape(-1, self.n_agents)

        # switch, bridge samples
        state = F.sigmoid(state)
        state = (state >= 0.5).float()
        switch = state[:, :self.n_switches]
        rest_of_stuff = state[:, self.n_switches:]
        samples = torch.cat((switch, IRsamples, rest_of_stuff), axis=1)
        return samples

    def sample_for_rtilde(self, x):
        IRdist, state = self.forward(x)
        IR_mean = IRdist.mean
        # bug fix: calculate r_tilde from logits
        state = F.sigmoid(state)
        # rest_of_state = state.float()
        rest_of_state = (state >= 0.5).float()
        return IR_mean, rest_of_state

class StateNetwork(nn.Module):
    def __init__(self, input_size, hidden_size=500, output_size=3, n_agents=3, n_switches=1, n_bridges=3):
        super().__init__()
        self.n_agents = n_agents
        self.n_switches = n_switches
        self.n_bridges = n_bridges
        self.output_size = output_size

        if n_bridges > 0:
            self.state_size = n_switches + (n_agents + 1) + (n_agents) + (n_bridges + 1) * n_agents
            self.action_size = (2 * n_switches + 2 + 3) * self.n_agents
        else:
            self.state_size = n_switches + n_agents * 2
            self.action_size = n_agents * (2 * n_switches + 2)


        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        x = x.reshape(-1, self.state_size + self.action_size)
        return self.network(x)


class CategoricalNetworks(nn.Module):
    def __init__(self, in_dim, hidden_size=500, out_dim=3, n_agents=3, n_switches=1, n_bridges=3):
        super().__init__()
        self.n_agents = n_agents
        self.n_switches = n_switches
        self.n_bridges = n_bridges

        if n_bridges > 0:
            self.state_size = n_switches + (n_agents + 1) + (n_agents) + (n_bridges+1)*n_agents # does not include last action
            self.action_size = (2 * n_switches + 2 + 3) * self.n_agents
        else:
            self.state_size = n_switches + n_agents * 2
            self.action_size = n_agents * (2 * n_switches + 2)

        self.out_dim = out_dim

        hidden_size=500
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
            nn.Softmax()
        )
    def forward(self, x):
        x = x.reshape(-1, self.state_size + self.action_size)

        params = self.network(x).reshape(-1, self.out_dim)
        # fix
        # return OneHotCategorical(logits=params)
        return OneHotCategorical(probs=params)


def evaluate(dataset_loader, net, model_type, criterion):
    with torch.no_grad():
        running_loss = 0
        for i, data in enumerate(dataset_loader):
            inputs, labels = data
            if model_type == "state_state":
                loss = net.loss(inputs, labels).mean()
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            running_loss += loss.item()
        return running_loss / len(dataset_loader)


def train(model_type, t_env, epochs=500, learning_rate=1e-3, batch_size=100, hidden_size=500, debug=False,
          env_model_directory="models_all", reward_type='category', model_num=-1, n_agents=3, n_switches=1,
          episode_limit=0, print_progress=False, n_bridges=3):
    if (model_num==-1 and model_type =="state_state"):
        raise ValueError("model_num should not be -1")
    prefix = working_directory+"/"+env_model_directory+"/"+str(t_env)
    # prefix = os.path.join(working_directory, env_model_directory, str(t_env))
    print("loading dataset from %s" % prefix)
    sys.stdout.flush()

    # get dataset
    dataset = np.load(prefix+"/dataset.npy")
    np.random.shuffle(dataset)
    dataset_train = dataset[:int(len(dataset) * 0.7)]
    dataset_val = dataset[int(len(dataset) * 0.7):]
    print("dataset shape: ", dataset.shape)
    sys.stdout.flush()

    possible_rewards = [0, 1, -1]
    x_train, y_train = preprocess(dataset_train, model_type=model_type, possible_rewards=possible_rewards,
                                  reward_type=reward_type, n_agents=n_agents, n_switches=n_switches, n_bridges=n_bridges)
    x_val, y_val = preprocess(dataset_val, model_type=model_type, possible_rewards=possible_rewards,
                              reward_type=reward_type, n_agents=n_agents, n_switches=n_switches, n_bridges=n_bridges)

    print("preprocess ******************", x_train.shape, y_train.shape)
    dt_train = TensorDataset(x_train, y_train)
    dt_val = TensorDataset(x_val, y_val)

    train_log = []
    val_log = []
    val_loss_min = 1e8

    input_size = x_train.shape[1]
    if model_type == "state_reward" and reward_type == "category":
        output_size = 3
    else:
        output_size = y_train.shape[1]

    if model_type == "state_state":
        net = DensityNetworkSwitch_state(input_size, hidden_size=hidden_size, n_agents=n_agents, n_switches=n_switches,
                                         n_bridges=n_bridges).to(device)
    else:
        net = Network(input_size, output_size, hidden_size, model_type, reward_type, switch=True).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    early_stop_threshold = 25
    early_stop_cur = 0

    train_loader = DataLoader(dt_train, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dt_val, shuffle=True, batch_size=batch_size)
    criterion = get_criterion(model_type, reward_type)
    if model_type == "state_obs":
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    for e in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if model_type == "state_state":
                loss = net.loss(inputs, labels).mean()
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)

            # import pdb; pdb.set_trace()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss /= len(train_loader)

        train_log.append(running_loss)
        val_loss = evaluate(val_loader, net, model_type, criterion)
        val_log.append(val_loss)

        if val_loss < val_loss_min:
            early_stop_cur = 0
            val_loss_min = val_loss

            if model_type == "state_reward":
                torch.save(net.state_dict(), '%s/%s_%s.pt'
                           % (prefix, model_type, reward_type))
            elif model_type == "state_state":
                torch.save(net.state_dict(), '%s/%s_%d.pt'
                           % (prefix, model_type, model_num))
            else:
                # print ("saved: ",'%s/%s.pt' % (prefix, model_type))
                torch.save(net.state_dict(), '%s/%s.pt'
                           % (prefix, model_type))
        else:
            early_stop_cur += 1

        if print_progress:
            print(running_loss)
            print(e + 1, ", ", i + 1)
            print("train loss:", running_loss)
            print('val loss: ', val_loss)

        if model_type == "state_reward":
            with open('%s/%s_%s_train_log.pkl'
                      % (prefix, model_type, reward_type), 'wb') as f:
                pickle.dump(train_log, f)
            with open('%s/%s_%s_val_log.pkl'
                      % (prefix, model_type, reward_type), 'wb') as f:
                pickle.dump(val_log, f)

        #ensemble method
        elif model_type == "state_state":
            with open('%s/%s_%d_train_log.pkl'
                      % (prefix, model_type, model_num), 'wb') as f:
                pickle.dump(train_log, f)
            with open('%s/%s_%d_val_log.pkl'
                      % (prefix, model_type, model_num), 'wb') as f:
                pickle.dump(val_log, f)

        else:
            with open('%s/%s_train_log.pkl'
                      % (prefix, model_type), 'wb') as f:
                pickle.dump(train_log, f)
            with open('%s/%s_val_log.pkl'
                      % (prefix, model_type), 'wb') as f:
                pickle.dump(val_log, f)

        if early_stop_cur >= early_stop_threshold:
            print("min val loss ", val_loss_min)
            print("finished training", model_type, "\n")
            sys.stdout.flush()
            return net

    sys.stdout.flush()
    print("min val loss", val_loss_min)
    print(model_type, " more training required? \n")
    return net


if __name__ == "__main__":

    try:
        mp.set_start_method('spawn')
    except:
        pass
    # directory = "test_ds10000-0-0_bs32_bsr1_targetupdate200_lr5.0E-04_epsilonaneal50000"
    # # directory = "testbridgemb10k"
    # # directory = "tmptest"
    # n_agents = 3
    # n_switches = 1
    # n_bridges = 1
    # processes = []


    # # for model_type in ["state_state", "state_obs", "state_done", "state_avail", "state_reward"]:
    # for model_type in ["state_avail"]:
    #     pargs = (model_type, 0,)
    #     pkwargs = {"epochs": 700, "learning_rate": 1e-3, "batch_size": 1000, "hidden_size": 500, "debug": True,
    #                "env_model_directory": directory, "model_num": 0, "n_agents": n_agents,
    #               "n_switches": n_switches, "n_bridges": n_bridges, "print_progress": False, }
    #     p = mp.Process(target=train, args=pargs, kwargs=pkwargs)
    #     processes.append(p)
    #     p.daemon = True
    #     p.start()
    #
    # for p in processes:
    #     p.join()
    #


    # directory = "test_ds10000-0-0_bs32_bsr1_targetupdate200_lr5.0E-04_epsilonaneal50000"
    # directory = "test10k3a1s3b"
    # directory = "testbridgemb10k"
    # directory = "tmptest"
    # directory = "testtest"
    directory = "test128"
    n_agents = 3
    n_switches = 1
    n_bridges = 3
    processes = []

    train("state_state", 0, batch_size=1000, learning_rate=1e-3, hidden_size=500, debug=True, print_progress=True,
          env_model_directory=directory, reward_type="category", model_num=0, n_switches=n_switches, n_agents=n_agents,
          n_bridges=n_bridges)

    # train("state_done", 0, batch_size=1000, hidden_size=500, debug=True,
    #       env_model_directory=directory, reward_type="category", model_num=0, n_switches=n_switches, n_agents=n_agents,
    #           n_bridges=n_bridges)

    # train("state_avail", 0, batch_size=100, hidden_size=500, debug=True,
    #       env_model_directory=directory, reward_type="category", model_num=0, n_switches=n_switches, n_agents=n_agents,
    #           n_bridges=n_bridges)

    # train("state_obs", 0, batch_size=1000, hidden_size=500, debug=True, env_model_directory=directory,
    #       reward_type="category", model_num=0, n_switches=n_switches, n_agents=n_agents,
    #       n_bridges=n_bridges)
    #
    # train("state_reward", 0, batch_size=1000, hidden_size=500, debug=True, env_model_directory=directory,
    #       reward_type="category", model_num=0, n_switches=n_switches, n_agents=n_agents,
    #           n_bridges=n_bridges)
