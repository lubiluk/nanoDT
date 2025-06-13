from pathlib import Path

import numpy as np
import optuna

import minari
import torch

from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries
import gymnasium as gym
from tqdm import tqdm

# Global settings
ENV_NAME = "LunarLanderContinuous-v3"
DATASET_NAME = f"{ENV_NAME}-expert-v0"  # Minari dataset name

SEED = 1234
N_TRIALS = 20
MAX_ITERS_PER_TRIAL = 5000
EVAL_INTERVAL = 1000

TRIAL_MODELS_DIR = Path("trials/")
TRIAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def eval_agent(trial: optuna.trial.Trial, agent: NanoDTAgent, reward_scale: float):
    env = gym.make(ENV_NAME)

    # TODO: Model cannot use 'act' method directly - it needs to be saved and loaded
    model_path = str(TRIAL_MODELS_DIR / f"trial_{trial.number}_model.pth")
    agent.save(model_path)

    agent = NanoDTAgent.load(model_path)
    returns = []

    for _ in tqdm(range(100)):
        agent.reset(target_return=reward_scale)
        obs, info = env.reset()
        done = False
        accumulated_rew = 0
        while not done:
            action = agent.act(obs)
            obs, rew, ter, tru, info = env.step(action)
            done = ter or tru
            accumulated_rew += rew
        returns.append(accumulated_rew)

    env.close()
    mean, std = np.mean(returns), np.std(returns)
    print("Mean return:", mean)
    return float(mean)


def objective(trial: optuna.trial.Trial) -> float:
    seed_libraries(SEED)
    minari_dataset = minari.load_dataset(DATASET_NAME)

    # Define search space
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    K = trial.suggest_categorical("K", [10, 20, 30, 40, 50, 60])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    reward_scale = trial.suggest_float("reward_scale", 10.0, 10000.0, log=True)

    # Agent's settings
    agent = NanoDTAgent(
        dropout=dropout,
        K=K,
        max_ep_len=1000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    agent.learn(
        minari_dataset,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        reward_scale=reward_scale,
        max_iters=MAX_ITERS_PER_TRIAL,
        eval_interval=EVAL_INTERVAL,
        warmup_iters=MAX_ITERS_PER_TRIAL // 10,
        lr_decay_iters=MAX_ITERS_PER_TRIAL,
        decay_lr=True  # LR Scheduling
    )

    return eval_agent(trial, agent, reward_scale)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_study.db",
        study_name="nanodt_hyperparam_tuning",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("Optimization finished")
    print("Best trial:")
    trial = study.best_trial

    print(f" Best mean return: {trial.value}")
    print("  Hyperparams:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
