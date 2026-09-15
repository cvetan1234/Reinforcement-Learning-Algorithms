import torch
import gymnasium as gym
import numpy as np
import optuna
import json
import time
from tqdm import tqdm
from model import DQN
from replay_buffer import ReplayBuffer
from config import *
from utils import preprocess, train_step

# Dictionary to store per-trial episode reward history
trial_rewards_dict = {}

def train_dqn(trial):
    """
    Trains a DQN agent on the CarRacing-v3 environment using hyperparameters
    sampled from an Optuna trial.

    Args:
        trial (optuna.trial.Trial): Optuna trial object providing hyperparameter suggestions.

    Returns:
        tuple: (mean_reward, rewards)
            mean_reward (float): Mean reward over the last 5 episodes.
            rewards (List[float]): Episode-wise reward history.
    """
    env = gym.make("CarRacing-v3", render_mode=None)
    action_dim = len(ACTIONS)

    # Initialize Q-network and target network
    q_net = DQN(action_dim).to(DEVICE)
    target_net = DQN(action_dim).to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())

    # Sample hyperparameters from the trial
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    epsilon_final = trial.suggest_float('epsilon_final', 0.01, 0.1)
    epsilon_decay = trial.suggest_int('epsilon_decay', 50000, 200000)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    target_update = trial.suggest_int('target_update', 500, 5000)

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(REPLAY_SIZE)

    # Define epsilon decay schedule
    epsilon_start = 1.0
    epsilon_by_frame = lambda f: epsilon_final + (epsilon_start - epsilon_final) * np.exp(-1. * f / epsilon_decay)

    rewards, durations = [], []
    frame_idx = 0
    pbar = tqdm(total=MAX_EPISODES, desc=f"Trial {trial.number}")

    for episode in range(MAX_EPISODES):
        start = time.time()
        obs, _ = env.reset()
        obs = preprocess(obs)
        total_reward = 0

        for step in range(MAX_STEPS):
            epsilon = epsilon_by_frame(frame_idx)
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(action_dim)
            else:
                with torch.no_grad():
                    state = torch.tensor(obs[None], dtype=torch.float32).to(DEVICE)
                    action_idx = int(torch.argmax(q_net(state), dim=1).item())

            action = np.array(ACTIONS[action_idx], dtype=np.float32)
            next_obs, reward, term, trunc, _ = env.step(action)
            next_obs = preprocess(next_obs)
            done = term or trunc

            # Store transition in replay buffer
            buffer.add((obs, action_idx, reward, next_obs, done))
            obs = next_obs
            total_reward += reward
            frame_idx += 1

            # Train if buffer is large enough
            if len(buffer) > batch_size:
                train_step(q_net, target_net, buffer, optimizer, gamma, batch_size, DEVICE)

            # Periodically update target network
            if frame_idx % target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

            if done:
                break

        rewards.append(total_reward)
        durations.append(time.time() - start)

        # Early prune slow trials
        if len(durations) >= 3 and np.mean(durations[-3:]) > EPISODE_TIMEOUT:
            print(f"Trial {trial.number} pruned (avg duration {np.mean(durations[-3:]):.2f}s > {EPISODE_TIMEOUT}s)")
            raise optuna.exceptions.TrialPruned()

        pbar.set_postfix(reward=total_reward)
        pbar.update(1)

        # Stop early if agent is performing well enought
        if len(rewards) >= 5 and np.mean(rewards[-5:]) > 500:
            break

    pbar.close()
    env.close()

    # Save model if it's the best so far
    mean_reward = np.mean(rewards[-5:])
    if not hasattr(train_dqn, "best_value") or mean_reward > train_dqn.best_value:
        train_dqn.best_value = mean_reward
        torch.save(q_net.state_dict(), "best_model.pth")

    return mean_reward, rewards

def objective(trial):
    """
    Optuna objective function to optimize DQN hyperparameters.

    Args:
        trial (optuna.trial.Trial): The current trial object.

    Returns:
        float: Mean reward over the last 5 episodes (used as trial score).
    """
    mean_reward, rewards = train_dqn(trial)
    trial_rewards_dict[trial.number] = rewards
    return mean_reward
