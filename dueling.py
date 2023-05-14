import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):# Initialize the network with input and output dimensions
        super(DuelingDQN, self).__init__()
         # Define the feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        # Define the advantage stream layer
        self.advantage_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        # Define the value stream layer
        self.value_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        features = self.feature_layer(x)# Apply the feature extraction layer to the input
        advantage = self.advantage_layer(features)# Apply the advantage stream layer to the extracted features
        value = self.value_layer(features)# Apply the value stream layer to the extracted features

        # Combine the advantage and value streams.
        try:
            # Combine value and advantage (correcting for the average advantage)
          q_values = value + (advantage - advantage.mean(dim=1, keepdim=True)) 
        except:
            # Error occurs probably because of a batch of size 1, use dimension 0 instead
          q_values = value + (advantage - advantage.mean(dim=0, keepdim=True))
          
        return q_values
