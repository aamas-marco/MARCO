import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import pickle
import os
import random
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

working_directory = '/home/MARCO'


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, model_type, reward_type, switch=False):
        super(Network, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        if model_type != "state_reward":
            self.l2_5 = nn.Linear(hidden_size, hidden_size)

        self.l3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

        self.model_type = model_type
        self.reward_type = reward_type
        self.switch = switch

    def forward(self, x):
        x = self.l1(x)
        x = self.tanh(x)
        x = self.l2(x)
        x = self.tanh(x)

        if self.model_type != "state_reward" and not self.switch:
            x = self.l2_5(x)
            x = self.tanh(x)

        x = self.l3(x)

        # try scaling output between 0 and 1
        if not self.switch and self.model_type == "state_state":
            pass

        elif self.switch and self.model_type == "state_obs":
            pass

        elif self.model_type == "state_done" or self.model_type == "state_avail":
            pass

        elif self.model_type == "state_reward" and self.reward_type == "category":
            pass

        elif self.model_type == "state_reward" and self.reward_type == "regression":
            x = self.softplus(x)

        elif self.model_type == "state_all":
            s_t, o_t, avail_t, done_t, r_t = split_model_all_out(x)
            if self.reward_type == "category":
                pass
            elif self.reward_type == "regression":
                r_t = self.softplus(r_t)
            x = torch.cat([s_t, o_t, avail_t, done_t, r_t], axis=-1)

        return x


def evaluate(dataset_loader, criterion, net):
    with torch.no_grad():
        running_loss = 0
        for i, data in enumerate(dataset_loader):
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        return running_loss / len(dataset_loader)


def split_model_all_out(vec):
    vec_s_t = vec[..., :48 - 27]
    vec_o_t = vec[..., 48 - 27:48 - 27 + 90]
    vec_avail_t = vec[..., 48 - 27 + 90:48 - 27 + 90 + 27]
    vec_done_t = vec[..., 48 - 27 + 90 + 27:48 - 27 + 90 + 27 + 1]
    vec_r_t = vec[..., 48 - 27 + 90 + 27 + 1:]

    return vec_s_t, vec_o_t, vec_avail_t, vec_done_t, vec_r_t

def get_criterion(model_type, reward_type):
    if model_type == "state_state" or model_type == "state_obs" or model_type == 'just_state':
        return torch.nn.MSELoss(reduction='mean')
    elif model_type == "state_reward" and reward_type == "regression":
        return torch.nn.MSELoss(reduction='mean')
    elif model_type == "state_reward" and reward_type == "category":
        return nn.CrossEntropyLoss(reduction='mean')
    elif model_type == "state_done" or model_type == "state_avail":
        return torch.nn.BCEWithLogitsLoss(reduction='mean')
    elif model_type == "state_all":
        def custom_loss(output, target):
            pred_s_t, pred_o_t, pred_avail_t, pred_done_t, pred_r_t = split_model_all_out(output)
            target_s_t, target_o_t, target_avail_t, target_done_t, target_r_t = split_model_all_out(target)
            l_s_t = F.mse_loss(pred_s_t, target_s_t)
            l_o_t = F.mse_loss(pred_o_t, target_o_t)
            l_avail_t = F.binary_cross_entropy_with_logits(pred_avail_t, target_avail_t)
            l_done_t = F.binary_cross_entropy_with_logits(pred_done_t, target_done_t)
            if reward_type == "regression":
                l_r_t = F.mse_loss(pred_r_t, target_r_t)
            elif reward_type == "category":
                l_r_t = F.cross_entropy(pred_r_t, target_r_t)
            return l_s_t + l_o_t + l_avail_t + l_done_t + l_r_t

        return custom_loss

    print("No valid model type")
    raise ValueError


def get_old_prefix(env_model_directory, t_env):
    timesteps = []
    tmp = os.path.join(working_directory, env_model_directory)
    for name in os.listdir(tmp):
        # Check if they are dirs the names of which are numbers
        if name.isdigit():
            if int(name) != t_env:
                timesteps.append(int(name))
    if len(timesteps) == 0:
        print("ERROR: where are the initial models??")
        raise ValueError
    last_timestep = max(timesteps)
    return last_timestep


def get_new_dataset(dataset_new, dataset_rnn_new=None, t_env=0, sampling_timesteps=100, env_model_directory="models_all",
                    policy_model_path="", epsilon=0.1, switch=False, exploration=False, n_agents=3):
    prefix_new = os.path.join(working_directory, env_model_directory, str(t_env))

    # np.random.shuffle(dataset_new)

    # make new directory
    if not os.path.exists(prefix_new):
        os.makedirs(prefix_new)

    # if this is the first dataset
    if t_env == 0:
        np.save("%s/dataset" % prefix_new, dataset_new)
        print("*************** SAVED ", "%s/dataset.npy" % prefix_new)
        if dataset_rnn_new is not None:
            np.save("%s/dataset_rnn" % prefix_new, dataset_rnn_new)
            print("*************** SAVED ", "%s/dataset_rnn.npy" % prefix_new)
        return None


    # load the most recent dataset collected previously
    last_timestep = get_old_prefix(env_model_directory, t_env)

    # merge the newly collected with the most recent dataset
    prefix_old = os.path.join(working_directory, env_model_directory, str(last_timestep))
    print ("******Merging dataset with %s" % prefix_old)

    dataset_old = np.load("%s/dataset.npy" % prefix_old)
    dataset_all = np.concatenate((dataset_old, dataset_new), axis=0)
    # save the merged dataset for purpose of collecting next dataset
    np.save("%s/dataset" % prefix_new, dataset_all)

    if dataset_rnn_new is not None:
        dataset_rnn_old = np.load("%s/dataset_rnn.npy" % prefix_old)
        dataset_rnn_all = np.concatenate((dataset_rnn_old, dataset_rnn_new), axis=0)
        # save the merged dataset for purpose of collecting next dataset
        np.save("%s/dataset" % prefix_new, dataset_rnn_all)

    return None

