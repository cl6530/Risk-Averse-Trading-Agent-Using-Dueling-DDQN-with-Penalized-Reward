import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )

        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        advantage = self.advantage_layer(features)
        value = self.value_layer(features)

        # Combine the advantage and value streams.
        try:
          q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        except:
          q_values = value + (advantage - advantage.mean(dim=0, keepdim=True))
          
        return q_values
