import optuna
from gridworld import *
from plot import *
import time
import numpy as np
import pandas as pd

TIMEOUT = 4*60*60       # Total optimization time in seconds
MAX_STEPS = 1000   # Max steps per episode

def value_iteration(env, gamma=1, theta=1e-4):
    number_steps, num_states, num_actions = 0, env.num_states(), env.num_actions()
    V, Q, policy = np.zeros(num_states), np.zeros((num_states, num_actions)), np.zeros((num_states, num_actions))
    while True:
        delta = 0
        for s in range(num_states):
            current_q_values = np.zeros(num_actions)
            for a in range(num_actions):
                obs, reward, done = env.step_dp(s, a)
                current_q_values[a] = reward + (gamma * V[obs] * (not done))
            Q[s] = current_q_values
            delta = max(delta, abs(V[s] - np.max(Q[s])))
            V[s] = np.max(current_q_values)
        if delta < theta:
            break
        number_steps += 1
    policy[np.arange(num_states), np.argmax(Q, axis=1)] = 1.0
    return V, Q, policy, number_steps

def epsilon_greedy_action(Q, state, epsilon):
    num_actions = Q.shape[1]
    return np.random.choice(num_actions) if np.random.rand() < epsilon else np.argmax(Q[state])

def sarsa(env, gamma=1, alpha=0.1, epsilon=0.1, episodes=1000, max_steps=1000):
    start_time = time.time()
    num_states, num_actions = env.num_states(), env.num_actions()
    Q = np.zeros((num_states, num_actions))
    early_stopped_count = 0

    for n in range(episodes):
        print(f"\r  ➤ SARSA Episode {n + 1}/{episodes}", end="", flush=True)

        state = env.reset()
        action = epsilon_greedy_action(Q, state, epsilon)
        done = False
        steps = 0

        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_action(Q, next_state, epsilon)
            Q[state, action] += alpha * (reward + gamma * Q[next_state, next_action] - Q[state, action])
            state, action = next_state, next_action
            steps += 1
            if steps >= max_steps:
                early_stopped_count += 1
                break

    print()  # newline after last episode print
    policy = np.eye(num_actions)[np.argmax(Q, axis=1)]
    V = np.max(Q, axis=1)
    elapsed_time = time.time() - start_time
    return V, Q, policy, elapsed_time, early_stopped_count

def q_learning(env, gamma=1, alpha=0.1, epsilon=0.1, episodes=1000, max_steps=1000):
    start_time = time.time()
    num_states, num_actions = env.num_states(), env.num_actions()
    Q = np.zeros((num_states, num_actions))
    early_stopped_count = 0

    for n in range(episodes):
        print(f"\r  ➤ Q-Learning Episode {n + 1}/{episodes}", end="", flush=True)

        state = env.reset()
        done = False
        steps = 0

        while not done:
            action = epsilon_greedy_action(Q, state, epsilon)
            next_state, reward, done = env.step(action)
            max_q_next = 0 if done else np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * max_q_next - Q[state, action])
            state = next_state
            steps += 1
            if steps >= max_steps:
                early_stopped_count += 1
                break

    print()  # newline after last episode print
    policy = np.eye(num_actions)[np.argmax(Q, axis=1)]
    V = np.max(Q, axis=1)
    elapsed_time = time.time() - start_time
    return V, Q, policy, elapsed_time, early_stopped_count

# Store trial results
trial_logs = []

def create_objective_with_timer(start_time, timeout):
    def objective(trial):
        elapsed = time.time() - start_time
        time_left = max(0, timeout - elapsed)
        print(f"\n⏱️  Time remaining: {int(time_left)}s")

        env = MazeWater2()
        gamma = trial.suggest_float("gamma", 0.90, 0.999)
        alpha = trial.suggest_float("alpha", 0.01, 0.5)
        epsilon = trial.suggest_float("epsilon", 0.01, 0.2)
        episodes = trial.suggest_int("episodes", 500, 20000, step=500)

        # Choose one:
        _, _, policy_learned, _, early_stops = sarsa(
            env, gamma, alpha, epsilon, episodes=episodes, max_steps=MAX_STEPS
        )
        # _, _, policy_learned, _, early_stops = q_learning(
        #     env, gamma, alpha, epsilon, episodes=episodes, max_steps=MAX_STEPS
        # )

        _, _, policy_viter, _ = value_iteration(env, gamma=0.95)
        error = np.sum(~np.all(policy_learned == policy_viter, axis=1))

        score = error * 100000 + episodes
        print(f"⚠️  Early-stopped episodes this trial: {early_stops}/{episodes}")
        print(f"📊 Score = {score} (Errors: {error}, Episodes: {episodes})")

        # Save trial info
        trial_logs.append({
            "trial": trial.number,
            "gamma": round(gamma, 3),
            "alpha": round(alpha, 3),
            "epsilon": round(epsilon, 3),
            "episodes": episodes,
            "errors": error
        })

        return score
    return objective

if __name__ == "__main__":
    env = EmptyWorld55()
    gamma = 0.95
    alpha = 0.1
    epsilon = 0.1
    episodes = 1000

    V_q, Q_q, policy_q, elapsed_q, early_stops_q = q_learning(env, gamma=gamma, alpha=alpha, epsilon=epsilon, episodes=episodes, max_steps=MAX_STEPS)
    print(f"\n✅ Q-Learning completed in {elapsed_q:.2f}s with {early_stops_q} early-stopped episodes.")

    # Plot the Q-table
    # plot_q_table(env, Q_q)
    plot_v_table(env, V_q, policy_q)

    V_vi, _, policy_vi, _ = value_iteration(env, gamma=gamma)
    errors = np.sum(~np.all(policy_q == policy_vi, axis=1))
    print(f"❗ Number of errors compared to Value Iteration policy: {errors}")
    plot_v_table(env, V_vi, policy_vi)



