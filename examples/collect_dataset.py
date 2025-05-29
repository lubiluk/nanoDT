from argparse import ArgumentParser

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from tqdm.auto import tqdm

from minari import DataCollector


def collect_dataset(env_name: str, expert_path: str, n_timesteps: int, seed: int):
    torch.manual_seed(seed)

    env = DataCollector(gym.make(env_name))
    agent = SAC.load(expert_path, env=env)

    done = True

    for i in tqdm(range(n_timesteps)):
        if done:
            obs, _ = env.reset(seed=seed)

        action, _ = agent.predict(obs)
        obs, rew, ter, tru, info = env.step(action)

        done = ter or tru

    dataset = env.create_dataset(
        dataset_id=f"{env_name}-expert-v0",
        algorithm_name="ExpertPolicy",
        # code_permalink="https://minari.farama.org/tutorials/behavioral_cloning",
        author="Paweł Gajewski",
        # author_email="lubiluk@gmail.com.org",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env-name", type=str, default="FetchReach-v2")
    parser.add_argument(
        "--expert-path", type=str, default="logs/sac/FetchReach-v2_3/FetchReach-v2.zip"
    )
    parser.add_argument("--n-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1337)

    args = parser.parse_args()

    collect_dataset(
        env_name=args.env_name,
        expert_path=args.expert_path,
        n_timesteps=args.n_timesteps,
        seed=args.seed,
    )
