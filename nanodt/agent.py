from minari import MinariDataset
import torch
import gymnasium.spaces as spaces

from nanodt.model import DecisionTransformer, DecisionTransformerConfig
from nanodt.trainer import DecisionTransformerTrainer, DecisionTransformerTrainerConfig


class NanoDTAgent:
    def __init__(
        self,
        n_layer: int = 3,
        n_head: int = 1,
        n_embd: int = 128,
        dropout: float = 0.1,
        bias: bool = False,
        K: int = 20,
        max_ep_len: int = 1000,
        state_dim: int = 1,
        act_dim: int = 1,
        act_discrete: bool = True,
        act_vocab_size: int = 1,
        act_tanh: bool = False,
        tanh_embeddings: bool = False,
        device="auto",
    ):
        self.model_config = DecisionTransformerConfig(
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
            K=K,
            max_ep_len=max_ep_len,
            state_dim=state_dim,
            act_dim=act_dim,
            act_discrete=act_discrete,
            act_vocab_size=act_vocab_size,
            act_tanh=act_tanh,
            tanh_embeddings=tanh_embeddings,
        )
        self.device = device

    def learn(self, dataset: MinariDataset, **kwargs):
        # Automatically infer the state_dim and act_dim from the arguments
        self.model_config.state_dim = dataset.observation_space.shape[0]
        self.model_config.act_dim = dataset.action_space.shape[0]
        self.model_config.act_discrete = isinstance(
            dataset.action_space, spaces.Discrete
        )
        self.model = DecisionTransformer(config=self.model_config)
        self.reward_scale_ = kwargs.get("reward_scale", 0.0)
        self.trainer_config = DecisionTransformerTrainerConfig(device=self.device, **kwargs)
        trainer = DecisionTransformerTrainer(
            self.model, dataset, config=self.trainer_config
        )
        trainer.train()
        self.dataset_stats_ = trainer.dataset_stats_

    def save(self, path):
        """Save the NanoDTAgent to a file"""
        if not path.endswith(".pth"):
            path += ".pth"

        torch.save(
            {
                "model_state_dict": self.model.to("cpu").state_dict(),
                "model_config": self.model_config,
                "trainer_config": self.trainer_config,
                "state_mean": self.dataset_stats_.state_mean,
                "state_std": self.dataset_stats_.state_std,
                "reward_scale": self.reward_scale_,
            },
            path,
        )

    @classmethod
    def load(cls, path, env=None, device="cpu"):
        """Load a NanoDTAgent from a file."""

        if not path.endswith(".pth"):
            path += ".pth"

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Extract the saved model configuration
        model_config = checkpoint["model_config"]

        # Reconstruct the agent object
        agent = cls(
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            n_embd=model_config.n_embd,
            dropout=model_config.dropout,
            bias=model_config.bias,
            K=model_config.K,
            max_ep_len=model_config.max_ep_len,
            state_dim=model_config.state_dim,
            act_dim=model_config.act_dim,
            act_discrete=model_config.act_discrete,
            act_vocab_size=model_config.act_vocab_size,
            act_tanh=model_config.act_tanh,
            tanh_embeddings=model_config.tanh_embeddings,
        )

        # Load the model state
        agent.model = DecisionTransformer(config=model_config)
        agent.model_config = model_config
        agent.model.load_state_dict(checkpoint["model_state_dict"])

        # Load additional attributes if available
        agent.trainer_config = checkpoint.get("trainer_config", None)
        agent.state_mean_ = checkpoint.get("state_mean", None)
        agent.state_std_ = checkpoint.get("state_std", None)
        agent.reward_scale_ = checkpoint.get("reward_scale", None)

        return agent

    def reset(self, target_return):
        # TODO: allow user to move between devices
        device = next(self.model.buffers()).device
        act_dtype = torch.long if self.model_config.act_discrete else torch.float32
        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        self._ep_states = torch.zeros((1, self.model_config.state_dim)).to(
            device=device, dtype=torch.float32
        )
        self._ep_actions = torch.zeros(
            (0, self.model_config.act_dim), device=device, dtype=act_dtype
        )
        self._ep_rewards = torch.zeros(0, device=device, dtype=torch.float32)
        self._ep_rtgs = torch.tensor(
            target_return, device=device, dtype=torch.float32
        ).reshape(1, 1)
        self._ep_tsteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        self._target_return = target_return
        self._t = 0

    def act(self, obs, rew=None):
        device = next(self.model.buffers()).device
        act_dim = self.model_config.act_dim
        state_dim = self.model_config.state_dim
        max_len = self.model_config.K
        state_mean = torch.tensor(self.state_mean_, dtype=torch.float32)
        state_std = torch.tensor(self.state_std_, dtype=torch.float32)
        target_return = self._target_return
        scale = self.reward_scale_
        # Update state from the last step
        self._ep_states = torch.cat(
            [self._ep_states, torch.tensor(obs, dtype=torch.float32).reshape(1, -1).to(device)], dim=0
        )
        # Update reward if it's not the first step
        if rew is not None:
            self._ep_rtgs = torch.cat(
                [
                    self._ep_rtgs,
                    torch.tensor(
                        target_return - rew / scale, device=device, dtype=torch.float32
                    ).reshape(1, 1),
                ],
                dim=1,
            )
            self._ep_rewards[-1] = rew.item()

        # Padding for the next one
        self._ep_actions = torch.cat(
            [self._ep_actions, torch.zeros((1, act_dim), device=device)], dim=0
        )
        self._ep_rewards = torch.cat([self._ep_rewards, torch.zeros(1, device=device)])

        # prepare the input
        states = self._ep_states.reshape(1, -1, state_dim)[:, -max_len:]
        actions = self._ep_actions.reshape(1, -1, act_dim)[:, -max_len:]
        rtgs = self._ep_rtgs.reshape(1, -1, 1)[:, -max_len:]
        tsteps = self._ep_tsteps.reshape(1, -1)[:, -max_len:]

        # pad all tokens to sequence length
        # Calculate masks
        masks = (
            torch.cat(
                [torch.zeros(max_len - states.shape[1]), torch.ones(states.shape[1])],
                dim=0,
            )
            .to(dtype=torch.bool, device=device)
            .reshape(1, -1)
        )

        # Pad tensors
        states = pad_tensor(
            states, max_len, pad_value=0, device=device, dtype=torch.float32
        )
        actions = pad_tensor(
            actions, max_len, pad_value=-1, device=device, dtype=torch.long
        )
        rtgs = pad_tensor(
            rtgs, max_len, pad_value=0, device=device, dtype=torch.float32
        )
        tsteps = pad_tensor(
            tsteps, max_len, pad_value=0, device=device, dtype=torch.long
        )

        # get the action
        with torch.no_grad():
            logits, _ = self.model(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rtgs.to(dtype=torch.float32),
                tsteps.to(dtype=torch.long),
                masks,
            )

        if self.model_config.act_discrete:
            action = torch.argmax(logits[0, -1, :])
        else:
            action = logits[0, -1, :]

        self._ep_tsteps = torch.cat(
            [
                self._ep_tsteps,
                torch.ones((1, 1), device=device, dtype=torch.long) * (self._t + 1),
            ],
            dim=1,
        )

        return action.cpu().numpy()


def pad_tensor(tensor, pad_length, pad_value=0, pad_dim=1, device=None, dtype=None):
    """
    Pads a tensor along the specified dimension to the given length.

    Args:
        tensor (torch.Tensor): The tensor to pad.
        pad_length (int): The target length after padding.
        pad_value (float or int): The value to use for padding.
        pad_dim (int): The dimension to pad (default: 1).
        device (torch.device): The device to move the tensor to.
        dtype (torch.dtype): The dtype to cast the tensor to.

    Returns:
        torch.Tensor: The padded tensor.
    """
    pad_shape = list(tensor.shape)
    pad_shape[pad_dim] = pad_length - tensor.shape[pad_dim]
    padding = torch.full(pad_shape, pad_value, device=device, dtype=tensor.dtype)
    return torch.cat([padding, tensor], dim=pad_dim).to(dtype=dtype, device=device)
