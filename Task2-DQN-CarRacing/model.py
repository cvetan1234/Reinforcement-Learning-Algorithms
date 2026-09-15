import torch.nn as nn

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) for image-based input (e.g., CarRacing-v3).
    Uses a convolutional neural network to map raw image observations
    to Q-values for each discrete action.

    Args:
        action_dim (int): Number of discrete actions available in the environment.
    """
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        return self.net(x / 255.0)
