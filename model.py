import torch
import torch.nn as nn

class DepressionMLP(nn.Module):
    def __init__(self, input_dim):
        super(DepressionMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Output for binary classification
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)  # Squeeze for BCEWithLogitsLoss
