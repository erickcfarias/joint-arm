import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
import numpy as np

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


def weight_reset(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        layer.weight.data.uniform_(*hidden_init(layer))


class DDPG_Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DDPG_Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc1_bn = nn.BatchNorm1d(fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        return torch.tanh(self.fc2(x))


class DDPG_Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256,
                 fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(DDPG_Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc1_bn = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action)
           pairs -> Q-values.
        """
        xs = F.leaky_relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


class ActorA2C(nn.Module):
    def __init__(self, state_size, act_size):
        super(ActorA2C, self).__init__()
        self.seed = torch.manual_seed(7)
        self.base = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU()
        )
        self.mu = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_size),
            nn.Tanh(),
        )

        self.var = nn.Sequential(
            nn.Linear(64, act_size),
            nn.Softplus()
        )

        self.logstd = nn.Parameter(torch.zeros(act_size))
        self.reset_parameters()

    def forward(self, state):
        return self.mu(self.base(state)), self.var(self.base(state))

    def reset_parameters(self):
        self.base.apply(weight_reset)
        self.var.apply(weight_reset)
        self.mu.apply(weight_reset)


class CriticA2C(nn.Module):

    def __init__(self, state_size):
        super(CriticA2C, self).__init__()
        self.seed = torch.manual_seed(7)
        self.value = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.reset_parameters()

    def forward(self, state):
        return self.value(state)

    def reset_parameters(self):
        self.value.apply(weight_reset)


class ModelPPO(nn.Module):

    def __init__(self, state_size, action_size, hid_size, seed):
        super(ModelPPO, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.Linear(state_size, hid_size)
        self.actor = nn.Linear(hid_size, action_size)
        self.relu = F.relu
        self.tanh = torch.tanh

    def forward(self, state):
        x = self.fc(state)
        x = self.relu(x)
        x = self.actor(x)
        x = self.tanh(x)

        return {'action': x}
