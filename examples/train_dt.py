import minari
import torch
import os
from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries


def get_device():
    if torch.backends.mps.is_available():
        return "mps"  # macOS GPU
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def train_dt():
    seed = 1234
    seed_libraries(seed)

    dataset_name = "Ant-v4-expert-v0"
    save_path = f"output/dt/minari-{dataset_name}.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    minari_dataset = minari.load_dataset(dataset_name)

    device = get_device()
    print(f"Using device: {device}")


    dt_agent = NanoDTAgent(device=device)
    dt_agent.learn(minari_dataset, reward_scale=1000.0)
    dt_agent.save(save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train_dt()
