import pickle

import minari

from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries, TrajectoryDataset


def train_dt():
    seed = 1234
    seed_libraries(seed)
    minari_dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")

    # unpickle cache/halfcheetah-meium-v2.pkl
    # with open("cache/halfcheetah-medium-v2.pkl", "rb") as f:
    #     pickle_dataset = pickle.load(f)
    #     torch_dataset = TrajectoryDataset(pickle_dataset)
    #     print(next(iter(torch_dataset)).observations)

    dt_agent = NanoDTAgent(device="mps")
    dt_agent.learn(minari_dataset, reward_scale=1000.0)
    dt_agent.save("output/dt/minari-halfcheetah-medium-v0.pth")


if __name__ == "__main__":
    train_dt()

# +ml GCCcore/13.2.0 Python/3.11.5
# +source .venv/bin/activate