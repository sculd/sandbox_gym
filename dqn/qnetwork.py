import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

class QNetwork(nn.Module):
    def __init__(self, state_size, action_space, seed):
        """
        Build a fully connected neural network
        
        Parameters
        ----------
        state_size (int): State dimension
        action_space: gym action space
        seed (int): random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.action_space = action_space
        self.fc3 = nn.Linear(64, action_space.n)

    def forward(self, x):
        """Forward pass"""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
