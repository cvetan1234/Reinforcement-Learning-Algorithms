import torch
import numpy as np
import random
from collections import deque
from config import DEVICE

class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.

    Stores past experiences (transitions) and allows sampling random mini-batches
    for training. This helps break temporal correlations and stabilizes learning.

    Each transition is a tuple of:
        (state, action, reward, next_state, done)

    Args:
        size (int): Maximum number of transitions to store.
    """

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = map(np.array, zip(*batch))

        # Convert each batch component to torch tensors and move to the correct device
        return (
            torch.tensor(obs, dtype=torch.float32).to(DEVICE),
            torch.tensor(actions, dtype=torch.long).to(DEVICE),
            torch.tensor(rewards, dtype=torch.float32).to(DEVICE),
            torch.tensor(next_obs, dtype=torch.float32).to(DEVICE),
            torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        )

    def __len__(self):
        return len(self.buffer)
