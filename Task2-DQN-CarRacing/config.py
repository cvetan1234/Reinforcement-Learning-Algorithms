import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Discrete action space used for CarRacing-v3, Format: [steering, acceleration, brake]
ACTIONS = [
    [0.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 0.8],
    [0.0, 0.0, 0.0],
]

REPLAY_SIZE = 100_000
MAX_EPISODES = 400
MAX_STEPS = 1000
TIMEOUT = 8 * 60 * 60
EPISODE_TIMEOUT = 40



