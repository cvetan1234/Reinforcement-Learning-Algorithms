import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gridworld import MazeWater2
from task1 import sarsa, q_learning
from plot import plot_v_table
from my_maze import MazeDFS


class Evaluator:
    """
    Evaluator class for running and comparing RL methods (SARSA or Q-learning)
    using previously optimized hyperparameters stored in an Excel file.

    Attributes:
        env_class (class): The environment class to instantiate for each run.
        excel_path (str): Path to the Excel file containing Optuna results.
        methods (list[str]): RL methods to evaluate (e.g., ['sarsa', 'q_learning']).
        num_runs (int): Number of independent evaluation runs to average.
        episodes_override (int or None): If set, overrides episode count from Excel.
        final_visualizations (dict): Stores (env, V, policy) for each method for later plotting.
    """

    def __init__(self, env_class, excel_path, methods, num_runs=10, episodes_override=None):
        """
        Initializes the Evaluator.

        Args:
           env_class (class): The environment class to use for evaluation.
           excel_path (str): Path to the Excel file with Optuna results.
           methods (list[str]): RL methods to evaluate.
           num_runs (int, optional): How many times to run each method. Defaults to 10.
           episodes_override (int, optional): Overrides the number of training episodes.
        """
        self.env_class = env_class
        self.excel_path = excel_path
        self.methods = methods
        self.num_runs = num_runs
        self.episodes_override = episodes_override
        self.final_visualizations = {}

    def get_best_params(self):
        """
        Loads the best hyperparameter set from the Excel file based on the highest score.

        Returns:
            dict: A dictionary of hyperparameters from the best trial.
        """
        df = pd.read_excel(self.excel_path)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        best_row = df.loc[df['score'].idxmax()]
        return best_row.to_dict()

    def evaluate_method(self, method, best_params):
        """
        Runs the specified method multiple times and averages the episode rewards.

        Args:
            method (str): The method to evaluate ('sarsa' or 'q_learning').
            best_params (dict): The best hyperparameters to use.

        Returns:
            np.ndarray: A 2D array (num_runs x episodes) of episode rewards.
        """
        all_episode_rewards = []
        final_V, final_policy, final_env = None, None, None

        print(f"\nUsing parameters for {method.upper()}:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        for run in range(self.num_runs):
            print(f"Running {method.upper()} - Iteration {run + 1}/{self.num_runs}")
            env = self.env_class()

            # Extract hyperparameters
            gamma = best_params["gamma"]
            alpha = best_params["alpha"]
            epsilon = best_params["epsilon"]
            lambda_ = best_params["lambda"]
            strategy = best_params["exploration_strategy"]
            optimistic_init = best_params["optimistic_init"]
            r_max_bonus = best_params["r_max_bonus"]
            c_ucb = best_params["c_ucb"]
            episodes = self.episodes_override or int(best_params["episodes"])
            max_steps = 1000

            # Run selected method
            if method == "sarsa":
                V, _, policy, _, _, _, episode_rewards = sarsa(
                    env, gamma, alpha, epsilon, lambda_, episodes, max_steps,
                    strategy, optimistic_init, r_max_bonus, c_ucb,
                    return_episode_rewards=True
                )
            elif method == "q_learning":
                V, _, policy, _, _, _, episode_rewards = q_learning(
                    env, gamma, alpha, epsilon, lambda_, episodes, max_steps,
                    strategy, optimistic_init, r_max_bonus, c_ucb,
                    return_episode_rewards=True
                )
            else:
                raise ValueError("Unknown method:", method)

            all_episode_rewards.append(episode_rewards)

            # Save final run's env, V, and policy for visualization
            if run == self.num_runs - 1:
                final_env = env
                final_V = V
                final_policy = policy

        # Store final env, V, and policy for later visualization
        self.final_visualizations[method] = (final_env, final_V, final_policy)

        return np.array(all_episode_rewards)

    def run(self):
        """
        Runs the full evaluation pipeline:
        1. Load best parameters
        2. Run each method num_runs times
        3. Plot average episode rewards
        4. Plot final value table and policy
        """
        best_params = self.get_best_params()
        plt.figure(figsize=(10, 6))

        for method in self.methods:
            rewards_matrix = self.evaluate_method(method, best_params)
            avg_rewards = rewards_matrix.mean(axis=0)
            plt.plot(avg_rewards, label=method.upper())

        plt.title(f"Average Episode Reward Over {self.num_runs} Runs")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot V-table after reward curves
        for method, (env, V, policy) in self.final_visualizations.items():
            print(f"\nPlotting V-table for {method.upper()}")
            plot_v_table(env, V, policy)


if __name__ == "__main__":
    """
    evaluator = Evaluator(
        env_class=MazeWater2,
        excel_path="optuna_q_learning_results.xlsx",
        methods=["sarsa", "q_learning"],
        num_runs=10,
        episodes_override=500
    )
    """

    evaluator = Evaluator(
        env_class=lambda: MazeDFS(size=11, seed=42),
        excel_path="optuna_q_learning_custom_maze_results_1.xlsx",
        methods=["sarsa", "q_learning"],
        num_runs=10,
        episodes_override=500
    )
    
    
    evaluator.run()
