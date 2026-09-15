from config import DEVICE, TIMEOUT
from train import objective, trial_rewards_dict
from evaluate import evaluate
import optuna
import pandas as pd
import json

# Track the best reward achieved across all Optuna trials
best_value_so_far = -float("inf")

def save_best_params_callback(study, trial):
    """
    Optuna callback function to save the best-performing trial's hyperparameters
    and optionally stop early if a reward threshold is met.

    Args:
        study (optuna.study.Study): The current Optuna study.
        trial (optuna.trial.FrozenTrial): The trial that just completed.
    """

    global best_value_so_far

    # Save new best trial
    if trial.value is None:
        return
    if trial.value > best_value_so_far:
        best_value_so_far = trial.value
        print(f"\n💾 Saving improved trial {trial.number} with reward {trial.value:.2f}")
        with open("best_params.json", "w") as f:
            json.dump(trial.params, f, indent=4)

    # Stop early if threshold met
    if trial.value > 500:
        print(f"\n🎉 Early stopping: Trial {trial.number} reached reward {trial.value:.2f}")
        study.stop()

if __name__ == "__main__":
    print(f"🖥️ Using device: {DEVICE}")

    # Choose optimization or evaluation
    RUN_OPTIMIZATION = True
    RUN_EVALUATION = False

    if RUN_OPTIMIZATION:
        # Create and optimize Optuna study
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, timeout=TIMEOUT, callbacks=[save_best_params_callback])

        # Print best hyperparameters found
        print("\n🔍 Best hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        # Save all trial results (including episode rewards)
        df = []
        for trial in study.trials:
            entry = {'Trial': trial.number, 'MeanReward': trial.value}
            entry.update(trial.params)
            for i, r in enumerate(trial_rewards_dict.get(trial.number, []), 1):
                entry[f"Episode {i}"] = r
            df.append(entry)

        pd.DataFrame(df).to_excel("optuna_results.xlsx", index=False)
        print("✅ Results saved to 'optuna_results.xlsx'")

    if RUN_EVALUATION:
        # Play the best saved model
        evaluate(render=True)
