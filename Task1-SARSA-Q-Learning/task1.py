from dependency_installer import DependencyInstaller

# Required packages
required = {
    "optuna": "optuna",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm"
}

DependencyInstaller(required).install_all()

import optuna
from gridworld import *
from plot import *
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from my_maze import MazeDFS

TIMEOUT = 6 * 60 * 60   # Maximum total tuning time (seconds)
MAX_STEPS = 1000    # Max steps per episode

def select_action(Q, state, num_actions, epsilon, total_steps, strategy, counts=None, c_ucb=1.0):
    """
    Selects an action based on the chosen exploration strategy.

    Args:
        Q (ndarray): Q-value table.
        state (int): Current state.
        num_actions (int): Total number of possible actions.
        epsilon (float): Exploration probability (for epsilon-based strategies).
        total_steps (int): Total steps taken so far (used for decay/UCB).
        strategy (str): Exploration method: "epsilon_greedy", "decay_epsilon_greedy", or "ucb1".
        counts (ndarray, optional): Visit counts for UCB1.
        c_ucb (float): Exploration coefficient for UCB1.

    Returns:
        int: Selected action.
    """
    
    if strategy == "epsilon_greedy":
        return np.random.choice(num_actions) if np.random.rand() < epsilon else np.argmax(Q[state])
    elif strategy == "decay_epsilon_greedy":
        decayed = max(epsilon / (1 + 0.0005 * total_steps), 0.01)
        return np.random.choice(num_actions) if np.random.rand() < decayed else np.argmax(Q[state])
    elif strategy == "ucb1":
        if counts is None or np.any(counts[state] == 0):
            return np.random.choice(np.where(counts[state] == 0)[0])
        ucb_vals = Q[state] + c_ucb * np.sqrt(np.log(total_steps + 1) / (counts[state] + 1e-8))
        return np.argmax(ucb_vals)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def sarsa(env, gamma, alpha, epsilon, lambda_, episodes, max_steps, strategy, optimistic_init, r_max_bonus, c_ucb, return_episode_rewards=False):
    """
    Runs SARSA(λ) algorithm with eligibility traces and optional exploration enhancements.

    Args:
        env (GridworldEnv): The environment.
        gamma (float): Discount factor.
        alpha (float): Learning rate.
        epsilon (float): Exploration probability.
        lambda_ (float): Trace decay parameter.
        episodes (int): Number of training episodes.
        max_steps (int): Max steps per episode.
        strategy (str): Exploration strategy.
        optimistic_init (float): Initial Q-values.
        r_max_bonus (float): Bonus for rarely visited states (R-Max-style).
        c_ucb (float): UCB1 exploration coefficient.

    Returns:
        tuple: (V, Q, policy, early_stopped_count, total_cumulative_reward, last_episode_reward)
    """

    # Initialize Q-table and visit counts
    num_states, num_actions = env.num_states(), env.num_actions()
    Q = np.full((num_states, num_actions), optimistic_init)
    counts = np.zeros((num_states, num_actions))
    total_steps, total_cumulative_reward = 0, 0
    early_stopped_count, last_episode_reward = 0, 0
    episode_rewards = []

    for n in range(episodes):
        print(f"\r  ➤ SARSA(λ) Episode {n + 1}/{episodes}", end="", flush=True)

        # Reset environment and eligibility traces
        state, E, done, steps = env.reset(), np.zeros((num_states, num_actions)), False, 0
        episode_reward = 0
        action = select_action(Q, state, num_actions, epsilon, total_steps, strategy, counts, c_ucb)    # Choose initial action

        while not done:
            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Add R-Max bonus
            bonus = r_max_bonus / (np.sqrt(counts[state, action] + 1)) if r_max_bonus > 0 else 0
            reward += bonus

            next_action = select_action(Q, next_state, num_actions, epsilon, total_steps, strategy, counts, c_ucb)

            # Compute TD error and update eligibility traces
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
            E[state, action] += 1
            Q += alpha * delta * E
            E *= gamma * lambda_

            # Update tracking variables
            counts[state, action] += 1
            total_steps += 1
            state, action = next_state, next_action
            steps += 1

            # Handle early stopping
            if steps >= max_steps:
                early_stopped_count += 1
                break

        total_cumulative_reward += episode_reward
        last_episode_reward = episode_reward
        episode_rewards.append(episode_reward)

    print()

    # Derive deterministic greedy policy from Q
    policy = np.eye(num_actions)[np.argmax(Q, axis=1)]

    return np.max(Q, axis=1), Q, policy, early_stopped_count, total_cumulative_reward, last_episode_reward, episode_rewards if return_episode_rewards else []

def q_learning(env, gamma, alpha, epsilon, lambda_, episodes, max_steps, strategy, optimistic_init, r_max_bonus, c_ucb, return_episode_rewards=False):
    """
    Runs Q-Learning(λ) algorithm with eligibility traces and optional exploration enhancements.

    Args:
        env (GridworldEnv): The environment.
        gamma (float): Discount factor.
        alpha (float): Learning rate.
        epsilon (float): Exploration probability.
        lambda_ (float): Trace decay parameter.
        episodes (int): Number of training episodes.
        max_steps (int): Max steps per episode.
        strategy (str): Exploration strategy.
        optimistic_init (float): Initial Q-values.
        r_max_bonus (float): Bonus for rarely visited states (R-Max-style).
        c_ucb (float): UCB1 exploration coefficient.

    Returns:
        tuple: (V, Q, policy, early_stopped_count, total_cumulative_reward, last_episode_reward)
    """

    # Initialize Q-table and visit counts
    num_states, num_actions = env.num_states(), env.num_actions()
    Q = np.full((num_states, num_actions), optimistic_init)
    counts = np.zeros((num_states, num_actions))
    total_steps, total_cumulative_reward = 0, 0
    early_stopped_count, last_episode_reward = 0, 0
    episode_rewards = []

    for n in range(episodes):
        print(f"\r  ➤ Q-Learning(λ) Episode {n + 1}/{episodes}", end="", flush=True)

        # Reset environment and eligibility traces
        state, E, done, steps = env.reset(), np.zeros((num_states, num_actions)), False, 0
        episode_reward = 0

        while not done:
            # Choose action using specified strategy
            action = select_action(Q, state, num_actions, epsilon, total_steps, strategy, counts, c_ucb)

            next_state, reward, done = env.step(action)
            episode_reward += reward

            # Add R-Max bonus
            bonus = r_max_bonus / (np.sqrt(counts[state, action] + 1)) if r_max_bonus > 0 else 0
            reward += bonus

            # Compute TD error
            delta = reward + gamma * np.max(Q[next_state]) - Q[state, action]
            E[state, action] += 1
            Q += alpha * delta * E
            E *= gamma * lambda_

            # Update tracking variables
            counts[state, action] += 1
            total_steps += 1
            state = next_state
            steps += 1

            # Handle early stopping
            if steps >= max_steps:
                early_stopped_count += 1
                break

        total_cumulative_reward += episode_reward
        last_episode_reward = episode_reward
        episode_rewards.append(episode_reward)

    print()

    # Derive deterministic greedy policy from Q
    policy = np.eye(num_actions)[np.argmax(Q, axis=1)]

    return np.max(Q, axis=1), Q, policy, early_stopped_count, total_cumulative_reward, last_episode_reward, episode_rewards if return_episode_rewards else []

def evaluate_trial(env, policy):
    """
    Evaluates a trained policy by running a single episode in the environment using greedy action selection.

    Returns:
        int: 100 if goal reached, or number of steps taken otherwise.
    """
    
    state = env.reset()
    visited = set()
    steps = 0

    while True:
        if state in visited:
            return steps    # loop detected
        visited.add(state)

        action = np.argmax(policy[state])
        state, reward, done = env.step(action)
        steps += 1

        if reward == env.g_reward:
            return 100  # reached goal
        if reward == env.o_reward:
            return steps    # fell into water 

    return steps

def objective(trial, start_time, timeout, trial_logs, env, method="sarsa"):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial (optuna.Trial): The current trial.
        start_time (float): Time when optimization started.
        timeout (float): Timeout in seconds.
        trial_logs (list): List to store trial result dictionaries.
        env (GridworldEnv): The environment instance.
        method (str): Either "sarsa" or "q_learning".

    Returns:
        float: Objective score to maximize.
    """

    print(f"\nTime remaining: {int(timeout - (time.time() - start_time))}s")

    # Hyperparameter search space
    gamma = trial.suggest_float("gamma", 0.90, 0.999)
    alpha = trial.suggest_float("alpha", 0.0001, 0.5)
    epsilon = trial.suggest_float("epsilon", 0.0001, 0.2)
    lambda_ = trial.suggest_float("lambda", 0.0, 1.0)
    episodes = trial.suggest_int("episodes", 10, 10000, step=10)
    strategy = trial.suggest_categorical("exploration_strategy", ["ucb1"])
    optimistic_init = trial.suggest_float("optimistic_init", 0.0, 5.0)
    r_max_bonus = trial.suggest_float("r_max_bonus", 0.0, 5.0)
    c_ucb = trial.suggest_float("c_ucb", 0.1, 5.0)

    # Train
    if method == "sarsa":
        _, _, policy, early_stops, _, last_reward, _ = sarsa(env, gamma, alpha, epsilon, lambda_, episodes, MAX_STEPS,
                                                          strategy, optimistic_init, r_max_bonus, c_ucb)
    elif method == "q_learning":
        _, _, policy, early_stops, _, last_reward, _ = q_learning(env, gamma, alpha, epsilon, lambda_, episodes, MAX_STEPS,
                                                               strategy, optimistic_init, r_max_bonus, c_ucb)
    else:
        raise ValueError("Invalid method. Choose 'sarsa' or 'q_learning'.")

    # Evaluate policy
    steps = evaluate_trial(env, policy)
    score = steps * 100000 - episodes

    print(f"⚠️  Early-stopped: {early_stops}/{episodes}")
    print(f"🎯 Last episode reward: {last_reward}")

    # Log results
    trial_logs.append({
        "trial": trial.number,
        "gamma": round(gamma, 3),
        "alpha": round(alpha, 3),
        "epsilon": round(epsilon, 3),
        "lambda": round(lambda_, 3),
        "exploration_strategy": strategy,
        "optimistic_init": round(optimistic_init, 3),
        "r_max_bonus": round(r_max_bonus, 3),
        "c_ucb": round(c_ucb, 3),
        "episodes": episodes,
        "steps": steps,
        "score": score
    })

    return score

if __name__ == "__main__":
    # Control which algorithm to run
    RUN_SARSA = False
    RUN_Q_LEARNING = True

    # Set environment (used by both algorithms)
    """ENV = MazeWater2()"""
    ENV = MazeDFS(size=11, seed=42)
    
    if RUN_SARSA:
        print("\n=== Starting SARSA(λ) Optimization ===")
        logs = []
        start = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, start, TIMEOUT, logs, ENV, method="sarsa"), timeout=TIMEOUT)
        pd.DataFrame(logs).to_excel("optuna_sarsa_results.xlsx", index=False)

    if RUN_Q_LEARNING:
        print("\n=== Starting Q-Learning(λ) Optimization ===")
        logs = []
        start = time.time()
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, start, TIMEOUT, logs, ENV, method="q_learning"), timeout=TIMEOUT)
        pd.DataFrame(logs).to_excel("optuna_q_learning_results.xlsx", index=False)




