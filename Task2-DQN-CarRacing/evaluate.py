import gymnasium as gym
import torch
import numpy as np
from model import DQN
from config import *
from utils import preprocess

def evaluate(render=True, episodes=5):
    """
    Evaluates a trained DQN agent on the CarRacing-v3 environment.

    Args:
        render (bool): Whether to render the environment visually.
                       Set to False for headless evaluation.
        episodes (int): Number of episodes to run for evaluation.

    Loads:
        - The trained DQN model from 'best_model.pth'.

    Prints:
        - The reward for each episode.
        - The average reward across all episodes.
    """

    env = gym.make("CarRacing-v3", render_mode="human" if render else None)

    # Initialize and load trained DQN model
    q_net = DQN(len(ACTIONS)).to(DEVICE)
    q_net.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    q_net.eval()

    total_rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        obs = preprocess(obs)
        total = 0

        for _ in range(MAX_STEPS):
            # Select action using the DQN
            with torch.no_grad():
                state = torch.tensor(np.expand_dims(obs, 0), dtype=torch.float32).to(DEVICE)
                action_idx = int(torch.argmax(q_net(state), dim=1).item())

            # Execute action in the environment
            action = np.array(ACTIONS[action_idx], dtype=np.float32)
            next_obs, reward, term, trunc, _ = env.step(action)

            # Preprocess the next observation
            obs = preprocess(next_obs)
            total += reward

            # Stop the episode if terminated or truncated
            if term or trunc:
                break

        print(f"Episode {ep+1}: Reward = {total:.2f}")
        total_rewards.append(total)

    env.close()
    print(f"Average reward: {np.mean(total_rewards):.2f}")
