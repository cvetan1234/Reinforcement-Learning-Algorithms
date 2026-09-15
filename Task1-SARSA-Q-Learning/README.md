# Reinforcement Learning – Task 1

## Overview

This project implements and evaluates **SARSA(λ)** and **Q-Learning(λ)** reinforcement learning algorithms for gridworld environments.

It includes automatic hyperparameter optimization using **Optuna** and supports both the provided **MazeWater2** environment and a custom maze environment generated using **Depth-First Search (DFS)** (`MazeDFS`).

## Features

- SARSA(λ)
- Q-Learning(λ)
- Eligibility traces
- Automatic hyperparameter optimization with Optuna
- Evaluation of trained agents
- Support for the MazeWater2 environment
- Custom DFS-generated maze environment (`MazeDFS`)
- Storage of Optuna optimization results in Excel files
- Optional value-table visualization

## Requirements

- Python 3

Required Python libraries are automatically installed through:

```text
dependency_installer.py
```

## Project Files

```text
├── task1.py
├── evaluation.py
├── my_maze.py
├── dependency_installer.py
├── config.py
├── gridworld.py
├── playground.py
├── plot.py
├── optuna_q_learning_results.xlsx
└── optuna_q_learning_custom_maze_results_1.xlsx
```

### Main Files

- `task1.py` – runs the reinforcement learning training and Optuna hyperparameter optimization
- `evaluation.py` – evaluates the reinforcement learning agent
- `my_maze.py` – implements the custom DFS-generated `MazeDFS` environment
- `dependency_installer.py` – installs required dependencies
- `config.py` – project configuration
- `gridworld.py` – gridworld environment functionality
- `playground.py` – supporting environment functionality
- `plot.py` – plotting functionality, including value-table visualization
- `optuna_q_learning_results.xlsx` – Optuna results for MazeWater2
- `optuna_q_learning_custom_maze_results_1.xlsx` – Optuna results for MazeDFS

## How to Run

Open a terminal in the project directory.

### MazeWater2

To run hyperparameter optimization:

```bash
python task1.py
```

To run the evaluation:

```bash
python evaluation.py
```

### Custom MazeDFS Environment

To use the custom `MazeDFS` environment:

1. Open `evaluation.py`.
2. Scroll to the `if __name__ == "__main__"` section.
3. Comment out the block that uses `MazeWater2`.
4. Uncomment the block for `MazeDFS`.
5. Make the corresponding change in `task1.py` where the environment is initialized.
6. Run the training and evaluation commands as usual:

```bash
python task1.py
python evaluation.py
```

This ensures that both training and evaluation use the custom maze environment.

## Results

Optuna hyperparameter optimization results are stored in:

```text
optuna_q_learning_results.xlsx
optuna_q_learning_custom_maze_results_1.xlsx
```

These contain the optimization results for the MazeWater2 and MazeDFS environments respectively.
