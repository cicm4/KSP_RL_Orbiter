import torch
import torch.nn as nn


class QVNModel(nn.Module):
    def __init__(self, n_obs, n_actions, hidden_size):
        super().__init__()
        # The critic scores state-action pairs directly.
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observation, action):
        return self.net(torch.cat([observation, action], dim=-1))
