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

from envs.mpe_env import make_env
from learn_model import get_new_dataset, Network, get_criterion

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')


working_directory = '/home/MARCO'
assert os.path.isdir(working_directory) == True


env = make_env("simple_reference")

def preprocess(dataset, model_type='state_state', possible_rewards=[], reward_type='category', n_agents=3, bridge=False):

    obs_size = 21*2
    action_size = 50*2
    if not bridge:
        state_size = 42
    else:
        state_size = 46
    n_agents = 2


    s_t = dataset[:, 0:state_size]
    s_tplus1 = dataset[:, state_size:state_size + state_size]
    a_t = dataset[:, state_size + state_size:state_size + state_size + n_agents]
    r_t = dataset[:, state_size + state_size + n_agents: state_size + state_size + n_agents + 1]
    avail_t = dataset[:, state_size + state_size + n_agents + 1:state_size + state_size + n_agents + 1 + action_size]
    o_t = dataset[:, state_size + state_size + n_agents + 1 + action_size: state_size + state_size + n_agents + 1 + action_size + obs_size]
    term = dataset[:, state_size + state_size + n_agents + 1 + action_size + obs_size:state_size + state_size + n_agents + 1 + action_size + obs_size + 1]
    # end_of_ep = dataset[:, state_size + state_size + n_agents + 1 + action_size + obs_size + 1:state_size + state_size + n_agents + 1 + action_size + obs_size + 1 + 1]
    # a_tminus1 = dataset[:, state_size + state_size + n_agents + 1 + action_size + obs_size + 1 + 1:state_size + state_size + n_agents + 1 + action_size + obs_size + 1 + 1 + action_size]
    s_tminus1 = dataset[:, state_size + state_size + n_agents + 1 + action_size + obs_size + 1:state_size + state_size + n_agents + 1 + action_size + obs_size + 1 + n_agents]
    # s_tminus1 = dataset[:, state_size + state_size + n_agents + 1 + action_size + obs_size + 1 + 1 + action_size:state_size + state_size + n_agents + 1 + action_size + obs_size + 1 + 1 + action_size + state_size]

    # s_t -> s_t+1
    if model_type == 'just_state':
        x = torch.tensor(s_t).float().to(device)
        y = torch.torch.tensor(s_tplus1).float().to(device)

    # s_t , a_t -> s_t+1
    elif model_type == 'state_state':
        x_s_t = torch.tensor(s_t).float().to(device)
        x_a_t = F.one_hot(torch.tensor(a_t, dtype=int).flatten(), num_classes=50).float().to(device)
        x_a_t = x_a_t.reshape(-1, 50 * n_agents)
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
        x_a_t = F.one_hot(torch.tensor(a_t, dtype=int).flatten(), num_classes=50).float().to(device)
        x_a_t = x_a_t.reshape(-1, 50 * n_agents)
        # print (x_state_last.shape, x_state.shape, x_act.shape)
        x = torch.cat((x_s_t, x_s_tplus1, x_a_t), axis=1)

        if reward_type == 'category':
            # todo some 0 and 1
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
        # non_end_indices = np.where(end_of_ep.flatten() == 0)[0]
        # filter out end_of_ep
        # print ("before take: **************", s_tplus1.shape, a_t.shape)
        # x = np.take(s_tplus1, non_end_indices, 0) # TODO
        # x = np.take(s_t, non_end_indices, 0)
        # x_a_t = np.take(a_t, non_end_indices, 0)

        # y = np.take(term, non_end_indices, 0)
        x_a_t = F.one_hot(torch.tensor(x_a_t, dtype=int).flatten(), num_classes=50).float().to(device)
        x_a_t = x_a_t.reshape(-1, 50 * n_agents)

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


class NetworkMPE(nn.Module):
    '''
    state to state
    '''
    def __init__(self, input_size, hidden_size=1000, n_agents=2, bridge=False):
        super().__init__()
        self.n_agents = n_agents
        self.bridge = bridge
        if not bridge:
            self.state_size = 42
        else:
            self.state_size = 46
        self.action_size = 100

        self.commNet = IRNetworks(input_size, hidden_size=hidden_size, out_dim=20, n_agents=self.n_agents)
        if not bridge:
            self.stateNet = Network(input_size, 16, hidden_size, "state_state", "regression", mpe=True)
        else:
            self.stateNet = Network(input_size, 20, hidden_size, "state_state", "regression", mpe=True)
        self.goalNet = IRNetworks(input_size, hidden_size=hidden_size, out_dim=6, n_agents=self.n_agents)

    def forward(self, x):
        x = x.reshape(-1, self.state_size + self.action_size)
        c1, c2 = self.commNet(x)
        g1, g2 = self.goalNet(x)
        return c1, c2, g1, g2, self.stateNet(x)

    def loss(self, x, y):
        comm_dist_0, comm_dist_1, g1, g2, state_vals = self.forward(x)
        y = y.reshape(-1, self.state_size)


        # y_comm_0 = y[:,11:21]
        # y_comm_1 = y[:,-10:]
        # y_g1 = (y[:,8:11] > 0.5).float()
        # y_g2 = (y[:,29:32] > 0.5).float()
        # y_state = torch.cat([y[:,:8], y[:,21:29]], dim=-1)
        # import pdb; pdb.set_trace()

        y_comm_0 = y[:,11:21]
        y_comm_1 = y[:,32:42]
        y_g1 = (y[:,8:11] > 0.5).float()
        y_g2 = (y[:,29:32] > 0.5).float()
        if not self.bridge:
            y_state = torch.cat([y[:,:8], y[:,21:29]], dim=-1)
        else:
            y_state = torch.cat([y[:,:8], y[:,21:29], y[:,-4:]], dim=-1)

        loss = -comm_dist_0.log_prob(y_comm_0).mean()
        loss = loss - comm_dist_1.log_prob(y_comm_1).mean()
        loss = loss - g1.log_prob(y_g1).mean()
        loss = loss - g2.log_prob(y_g2).mean()
        # import pdb; pdb.set_trace()
        loss = loss + F.mse_loss(state_vals, y_state)

        return loss

    def sample(self, x):
        comm_dist_0, comm_dist_1, g1, g2, state_vals = self.forward(x)
        comm_samples_0 = comm_dist_0.sample()
        comm_samples_1 = comm_dist_1.sample()
        g1 = g1.sample() * 0.5 + 0.25
        g2 = g2.sample() * 0.5 + 0.25
        # samples = torch.cat((state_vals[:,:8], g1, comm_samples_0, state_vals[:, 8:], g2, comm_samples_1),axis=1)
        samples = torch.cat((state_vals[:,:8], g1, comm_samples_0, state_vals[:, 8:16], g2, comm_samples_1, state_vals[:, 16:20]),axis=1)
        return samples




class BernoulliNetwork(nn.Module):
    def __init__(self, in_dim, hidden_size=500, output_size=4, n_agents=3):
        super().__init__()
        self.n_agents = n_agents
        self.state_size = 1 + 2 * self.n_agents
        self.action_size = self.n_agents * 4

        self.out_dim = output_size
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(-1, self.state_size + self.action_size)
        params = self.network(x).reshape(-1, self.out_dim)
        return Bernoulli(params)


class IRNetworks(nn.Module):
    def __init__(self, in_dim, hidden_size=500, out_dim=10, n_agents=2, bridge=False):
        super().__init__()
        self.n_agents = n_agents

        if not bridge:
            self.state_size = 42
        else:
            self.state_size = 46

        self.action_size = 100
        self.out_dim = out_dim

        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim),
        )

    def forward(self, x):
        x = x.reshape(-1, self.state_size + self.action_size)
        params = self.network(x)
        c1 = F.softmax(params[:,:self.out_dim // 2])
        c2 = F.softmax(params[:,self.out_dim // 2:])
        return OneHotCategorical(c1), OneHotCategorical(c2)



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
        if model_type == "state_obs":
            output = net(inputs[0]).cpu().data.numpy()
            # y_pred = (output >= 0.5).reshape(3, 3).astype(int)
            # print("prev state:", inputs[0][:4], "prediction: ", y_pred, "  label: ", labels[0])
        # return running_loss / len(dataset_loader)
        return running_loss


def train(model_type, t_env, epochs=1000, learning_rate=1e-3, batch_size=5000, hidden_size=500, debug=False,
          env_model_directory="models_all", reward_type='category', model_num=-1, n_agents=3, bridge=False):
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
                                  reward_type=reward_type, n_agents=n_agents, bridge=bridge)
    x_val, y_val = preprocess(dataset_val, model_type=model_type, possible_rewards=possible_rewards,
                              reward_type=reward_type, n_agents=n_agents, bridge=bridge)

    # print ("preprocess ******************", x_train.shape, y_train.shape)
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
        net = NetworkMPE(input_size, hidden_size=hidden_size, n_agents=n_agents, bridge=bridge).to(device)
    else:
        net = Network(input_size, output_size, hidden_size, model_type, reward_type, mpe=True).to(device)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    early_stop_threshold = 25
    early_stop_cur = 0

    train_loader = DataLoader(dt_train, shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dt_val, shuffle=True, batch_size=batch_size)
    criterion = get_criterion(model_type, reward_type)
    if model_type == "state_obs":
        criterion = torch.nn.BCELoss(reduction='mean')

    for e in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if model_type == "state_state":
                loss = net.loss(inputs, labels).mean()
            else:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # running_loss /= len(train_loader)

        train_log.append(running_loss)
        val_loss = evaluate(val_loader, net, model_type, criterion)
        val_log.append(val_loss)

        if val_loss < val_loss_min:
            early_stop_cur = 0
            val_loss_min = val_loss

            if model_type == "state_reward":
                torch.save(net.state_dict(), '%s/%s_%s.pt'
                           % (prefix, model_type, model_num))
            elif model_type == "state_state":
                torch.save(net.state_dict(), '%s/%s_%d.pt'
                           % (prefix, model_type, model_num))
            else:
                # print ("saved: ",'%s/%s.pt' % (prefix, model_type))
                torch.save(net.state_dict(), '%s/%s.pt'
                           % (prefix, model_type))
        else:
            early_stop_cur += 1

        if debug:
            print(running_loss)
            print(e + 1, ", ", i + 1)
            print("train loss:", running_loss)
            print('val loss: ', val_loss)

        if model_type == "state_reward":
            with open('%s/%s_%s_train_log.pkl'
                      % (prefix, model_type, model_num), 'wb') as f:
                pickle.dump(train_log, f)
            with open('%s/%s_%s_val_log.pkl'
                      % (prefix, model_type, model_num), 'wb') as f:
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

def test(directory="mbmpe_1b0.0_samplingt20000_relearnint2000000_ensemble1_epsilon0.1_reg"):
    # train("state_state", 0, batch_size=100, hidden_size=500, debug=True,
    #       env_model_directory=directory, reward_type="regression", model_num=0)

    # train("state_done", 0, batch_size=100, hidden_size=100, debug=True,
    #       env_model_directory=directory, reward_type="regression", model_num=0)

    # train("state_avail", 0, batch_size=100, hidden_size=100, debug=True,
    #       env_model_directory=directory, reward_type="category", model_num=0)

    # train("state_obs", 0, batch_size=100, hidden_size=100, debug=True, env_model_directory=directory,
    #       reward_type="category", model_num=0)

    train("state_reward", 0, hidden_size=500, debug=True, env_model_directory=directory,
          reward_type="regression", model_num=0)

if __name__ == "__main__":

    test(directory="mbmpe_1b0.0_samplingt20000_relearnint2000000_ensemble1_epsilon0.1_reg")
    # for i in range(3, 10):
    #     test(directory="switch_mb_rand_"+str(i*50)+"dataset")