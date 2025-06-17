import minari
import tqdm

from nanodt.agent import NanoDTAgent
from nanodt.utils import seed_libraries
from nanodt.trainer import Callback, DecisionTransformerTrainerConfig, TrainingLogs, EvaluationLogs


class TqdmCallback(Callback):
    pbar: tqdm.tqdm

    def on_train_begin(self, config: DecisionTransformerTrainerConfig):
        self.pbar = tqdm.tqdm(total=config.max_iters, desc="Training Decision Transformer")

    def on_log(self, logs: TrainingLogs, step: int):
        self.pbar.set_postfix({"loss": f"{logs['loss']:.4f}", "mfu": f"{logs['mfu'] * 100:.2f}%"})
        self.pbar.update(step - self.pbar.n)

    def on_evaluate(self, logs: EvaluationLogs, step: int):
        self.pbar.set_description(f"Training (val_loss: {logs['val_loss']:.4f})")

    def on_train_end(self):
        self.pbar.close()


def train_dt():
    seed = 1234
    seed_libraries(seed)
    minari_dataset = minari.load_dataset("mujoco/halfcheetah/medium-v0")

    dt_agent = NanoDTAgent(device="mps")
    dt_agent.learn(
        minari_dataset,
        reward_scale=1000.0,
        callback=TqdmCallback()
    )
    dt_agent.save("output/dt/minari-halfcheetah-medium-v0.pth")


if __name__ == "__main__":
    train_dt()
