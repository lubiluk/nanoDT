import os
import sys
from typing import Dict

import gymnasium as gym
import minari
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from minari import DataCollector
from rl_zoo3.train import train
from stable_baselines3 import PPO
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from gymnasium.spaces import flatdim, flatten
import numpy.typing as npt


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, dg_dim, ag_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + dg_dim + ag_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, act_dim)

    def forward(self, obs, dg, ag):
        x = torch.cat([obs, dg, ag], dim=-1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, observation: Dict[str, npt.ArrayLike]):
        obs = torch.as_tensor(observation["observation"], dtype=torch.float32)
        dg = torch.as_tensor(observation["desired_goal"], dtype=torch.float32)
        ag = torch.as_tensor(observation["achieved_goal"], dtype=torch.float32)

        with torch.no_grad():
            return self(obs, dg, ag).cpu().numpy()


def collate_fn(batch):
    return {
        "id": torch.Tensor([x.id for x in batch]),
        # "seed": torch.Tensor([x.seed for x in batch]),
        # "total_steps": torch.Tensor([x.total_steps for x in batch]),
        "observations": torch.nn.utils.rnn.pad_sequence(
            [
                torch.as_tensor(x.observations["observation"], dtype=torch.float32)
                for x in batch
            ],
            batch_first=True,
        ),
        "desired_goals": torch.nn.utils.rnn.pad_sequence(
            [
                torch.as_tensor(x.observations["desired_goal"], dtype=torch.float32)
                for x in batch
            ],
            batch_first=True,
        ),
        "achieved_goals": torch.nn.utils.rnn.pad_sequence(
            [
                torch.as_tensor(x.observations["achieved_goal"], dtype=torch.float32)
                for x in batch
            ],
            batch_first=True,
        ),
        "actions": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.actions, dtype=torch.float32) for x in batch],
            batch_first=True,
        ),
        "rewards": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.rewards) for x in batch], batch_first=True
        ),
        "terminations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.terminations) for x in batch], batch_first=True
        ),
        "truncations": torch.nn.utils.rnn.pad_sequence(
            [torch.as_tensor(x.truncations) for x in batch], batch_first=True
        ),
    }


def train_bc():
    minari_dataset = minari.load_dataset("FetchReach-v2-expert-v0")
    dataloader = DataLoader(
        minari_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn
    )

    env = minari_dataset.recover_environment()
    observation_space = env.observation_space
    action_space = env.action_space
    assert isinstance(observation_space, spaces.Dict)
    assert isinstance(action_space, spaces.Box)

    obs_dim = observation_space["observation"].shape[0]
    dg_dim = observation_space["desired_goal"].shape[0]
    ag_dim = observation_space["achieved_goal"].shape[0]
    act_dim = action_space.shape[0]

    policy_net = PolicyNetwork(obs_dim, dg_dim, ag_dim, act_dim)
    optimizer = torch.optim.Adam(policy_net.parameters())
    loss_fn = nn.MSELoss()

    num_epochs = 32

    for epoch in range(num_epochs):
        for batch in dataloader:
            a_pred = policy_net(
                batch["observations"][:, :-1],
                batch["desired_goals"][:, :-1],
                batch["achieved_goals"][:, :-1],
            )
            a_hat = batch["actions"]
            loss = loss_fn(a_pred, a_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}/{num_epochs}, Loss: {loss.item()}")


    # Save the model
    torch.save(policy_net.state_dict(), "output/bc/policy_net.pth")

    env = gym.make("FetchReach-v2")

    for _ in range(5):
        obs, _ = env.reset()
        done = False
        accumulated_rew = 0
        while not done:
            action = policy_net.act(obs)
            obs, rew, ter, tru, _ = env.step(action)
            done = ter or tru
            accumulated_rew += rew

        print("Accumulated rew: ", accumulated_rew)

    env.close()


if __name__ == "__main__":
    train_bc()
