import pickle

import minari

from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries


def train_dt():
    seed = 1234
    seed_libraries(seed)
    minari_dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")

    dt_agent = NanoDTAgent(device="mps")
    dt_agent.learn(minari_dataset, reward_scale=1000.0)
    dt_agent.save("output/dt/minari-halfcheetah-medium-v0.pth")


if __name__ == "__main__":
    train_dt()
