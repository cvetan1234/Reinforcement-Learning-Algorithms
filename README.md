# Reinforcement Learning Algorithms

## Overview

This repository contains two reinforcement learning projects covering both **classical reinforcement learning** and **deep reinforcement learning**.

The first project implements **SARSA(λ)** and **Q-Learning(λ)** for gridworld and maze environments, including eligibility traces and Optuna-based hyperparameter optimization.

The second project applies **Deep Q-Learning (DQN)** to the **CarRacing-v3** environment, using a neural network, experience replay, model evaluation, and Optuna-based hyperparameter optimization.

Together, the two tasks demonstrate the progression from tabular reinforcement learning methods to deep reinforcement learning for visual control.

## Projects

### Task 1 – SARSA(λ) & Q-Learning(λ) with Hyperparameter Optimization

Task 1 implements and evaluates **SARSA(λ)** and **Q-Learning(λ)** agents in gridworld environments.

Main topics include:

- SARSA(λ)
- Q-Learning(λ)
- Eligibility traces
- Optuna hyperparameter optimization
- Agent evaluation
- MazeWater2 environment
- Custom DFS-generated maze environment (`MazeDFS`)
- Value-table visualization
- Storage and analysis of optimization results

The project supports both the provided MazeWater2 environment and a custom maze environment generated using **Depth-First Search (DFS)**.

For detailed setup, execution, and file information, see:

```text
Task1-SARSA-Q-Learning/README.md
```

### Task 2 – Deep Q-Learning (DQN) for CarRacing-v3

Task 2 implements a **Deep Q-Network (DQN)** agent for the **CarRacing-v3** environment.

Main topics include:

- Deep Q-Learning (DQN)
- Deep Q-Network architecture
- Experience replay
- Replay buffer
- Visual environment observations
- Optuna hyperparameter optimization
- Model checkpoint saving and loading
- Training and evaluation
- Visual evaluation of the trained agent
- Experiment result analysis

The training objective is to achieve a cumulative reward greater than **500**. The best-performing model and the Optuna optimization results are stored as part of the project.

For detailed setup, execution, and file information, see:

```text
Task2-DQN-CarRacing/README.md
```

## Repository Structure

```text
Reinforcement-Learning-Algorithms/
├── README.md
│
├── Task1-SARSA-Q-Learning/
│   ├── README.md
│   ├── task1.py
│   ├── evaluation.py
│   ├── my_maze.py
│   ├── dependency_installer.py
│   ├── config.py
│   ├── gridworld.py
│   ├── playground.py
│   ├── plot.py
│   └── ...
│
└── Task2-DQN-CarRacing/
    ├── README.md
    ├── main.py
    ├── train.py
    ├── evaluate.py
    ├── model.py
    ├── replay_buffer.py
    ├── config.py
    ├── utils.py
    ├── plot_trials.py
    └── ...
```

## Methods

The repository covers two major approaches to reinforcement learning.

### Tabular Reinforcement Learning

Task 1 uses tabular methods in which action values are explicitly stored and updated. SARSA and Q-Learning are extended with eligibility traces through **SARSA(λ)** and **Q-Learning(λ)**.

### Deep Reinforcement Learning

Task 2 replaces the tabular representation with a **Deep Q-Network**, allowing the agent to learn from the visual observations of the CarRacing environment. A replay buffer is used to store experiences and support more stable learning.

## Hyperparameter Optimization

Both projects use **Optuna** for automatic hyperparameter optimization.

The optimization results are stored with the individual projects and can be used to analyze and compare different training configurations.

## Technologies

Technologies and libraries used across the two projects include:

- Python
- PyTorch
- Gymnasium
- Optuna
- NumPy
- Pandas
- Matplotlib
- OpenPyXL

## Running the Projects

Each task has its own dependencies, configuration, and execution procedure.

See the individual README files for detailed instructions:

```text
Task1-SARSA-Q-Learning/README.md
Task2-DQN-CarRacing/README.md
```
