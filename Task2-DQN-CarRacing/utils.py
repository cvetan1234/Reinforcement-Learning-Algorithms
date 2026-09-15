import torch
import torch.nn as nn

def preprocess(obs):
    return obs.transpose(2, 0, 1)

def train_step(q_net, target_net, buffer, optimizer, gamma, batch_size, device):
    """
    Perform a single training step on a batch sampled from the replay buffer.

    This function computes the DQN loss (MSE between Q-values and TD targets),
    backpropagates the error, and updates the Q-network.

    Args:
        q_net (nn.Module): The main Q-network being trained.
        target_net (nn.Module): The target network used to compute target Q-values.
        buffer (ReplayBuffer): Experience replay buffer to sample training data from.
        optimizer (torch.optim.Optimizer): Optimizer used to update q_net parameters.
        gamma (float): Discount factor for future rewards.
        batch_size (int): Number of samples to use in this training step.
        device (torch.device): The device (CPU or CUDA) where computation occurs.
    """

    # Sample a batch of transitions from the buffer
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)    # Compute Q(s, a) for the current policy

    # Compute target Q-values using the target network and Bellman equation
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + gamma * next_q * (1 - dones)
    loss = nn.MSELoss()(q_values, target)   # Compute loss between predicted Q-values and targets

    # Backpropagate the loss and update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
