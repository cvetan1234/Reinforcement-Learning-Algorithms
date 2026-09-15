# Deep Q-Learning (DQN) for CarRacing-v3 with Hyperparameter Optimization

## Overview

This project implements a **Deep Q-Network (DQN)** agent for the **CarRacing-v3** reinforcement learning environment.

The agent uses deep reinforcement learning to learn driving behavior from the environment, while **Optuna** is used to automatically optimize the training hyperparameters. The training objective is to achieve a cumulative reward greater than **500**.

The project is implemented in Python with a modular structure separating the model architecture, training, evaluation, replay buffer, configuration, and utility functions.

## Features

- Deep Q-Learning (DQN)
- Deep Q-Network architecture
- Experience replay using a replay buffer
- Automatic hyperparameter optimization with Optuna
- Training and evaluation modes
- Model checkpoint saving and loading
- Visual evaluation of the trained agent
- Training result storage in Excel
- Optional plotting of learning curves

## Requirements

- Python 3.8+
- pip

Install the required packages with:

```bash
pip install -r requirements.txt
```

Alternatively, install the main dependencies manually:

```bash
pip install gymnasium torch optuna pandas matplotlib tqdm openpyxl
```

## Project Files

```text
├── main.py
├── train.py
├── evaluate.py
├── model.py
├── replay_buffer.py
├── config.py
├── utils.py
├── plot_trials.py
├── requirements.txt
├── optuna_results.xlsx
├── best_model.pth
└── bericht.pdf
```

### Main Files

- `main.py` – main entry point for training and evaluation
- `train.py` – training logic and Optuna integration
- `evaluate.py` – loads and evaluates the best trained model
- `model.py` – Deep Q-Network architecture
- `replay_buffer.py` – replay buffer used to stabilize learning
- `config.py` – central configuration parameters such as episode count and device
- `utils.py` – helper functions for training steps and preprocessing
- `plot_trials.py` – optional script for plotting learning curves
- `optuna_results.xlsx` – results from the Optuna hyperparameter search
- `best_model.pth` – saved best-performing model
- `bericht.pdf` – project report

## Training

The behavior of the program is controlled through the following flags in `main.py`:

```python
RUN_OPTIMIZATION
RUN_EVALUATION
```

To start the program:

```bash
python main.py
```

For training and hyperparameter optimization, configure the flags in `main.py` accordingly.

Training continues until either:

- the configured time limit is reached (default: 8 hours), or
- an agent achieves a cumulative reward greater than 500

The resulting optimization data and best model are saved as:

```text
optuna_results.xlsx
best_model.pth
```

## Evaluation

To evaluate the best trained model with visual rendering:

1. Open `main.py`.
2. Set:

```python
RUN_OPTIMIZATION = False
RUN_EVALUATION = True
```

3. Run:

```bash
python main.py
```

The evaluation runs **5 episodes** in render mode and prints the cumulative reward for each episode.

## Results

Hyperparameter optimization results are stored in:

```text
optuna_results.xlsx
```

The best trained DQN model is stored in:

```text
best_model.pth
```

The supplied plotting script can be used to visualize learning curves from the experiments.

## About

This project demonstrates the application of **deep reinforcement learning** to a visual control problem, including DQN training, experience replay, hyperparameter optimization, model evaluation, and experiment analysis.
