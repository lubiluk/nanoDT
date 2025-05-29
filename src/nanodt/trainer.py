"""
Training routine of a Decision Transformer Model.
Based on Andrej Karpathy's nanoGPT and original Decision Transformer code. 

References:
1) nanoGPT training loop implementation: https://github.com/karpathy/nanoGPT/blob/master/train.py
2) Original Decision Transformer implementation: https://github.com/kzl/decision-transformer
"""

import math
import inspect
from dataclasses import dataclass
import random
import time

from minari import MinariDataset
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class DecisionTransformerDataCollator:
    def __init__(
        self,
        state_mean,
        state_std,
        K=20,
        max_ep_len=1000,
        reward_scale=1000.0,
        act_discrete=False,
    ):
        self.state_mean = state_mean
        self.state_std = state_std
        self.K = K
        self.max_ep_len = max_ep_len
        self.reward_scale = reward_scale
        self.act_discrete = act_discrete

    def __call__(self, batch):
        max_len = self.K
        s, a, r, rtg, timesteps, mask = [], [], [], [], [], []
        state_dim = batch[0].observations.shape[-1]
        act_dim = batch[0].actions.shape[-1]
        max_ep_len = self.max_ep_len
        state_mean = self.state_mean
        state_std = self.state_std
        scale = self.reward_scale
        act_dtype = torch.long if self.act_discrete else torch.float32

        for traj in batch:
            si = random.randint(0, traj.rewards.shape[0] - 1)

            # get sequences from dataset
            # [:-1] is here to not use the last observation (one after the last action)
            # could there be more appropriate place for this cutoff?
            s.append(traj.observations[:-1][si : si + max_len].reshape(1, -1, state_dim))
            a.append(traj.actions[si : si + max_len].reshape(1, -1, act_dim))
            r.append(traj.rewards[si : si + max_len].reshape(1, -1, 1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            # TODO: Investigate this
            timesteps[-1][timesteps[-1] >= max_ep_len] = (
                max_ep_len - 1
            )  # padding cutoff
            rtg.append(
                self.discount_cumsum(traj.rewards[si:], gamma=1.0)[
                    : s[-1].shape[1]  # + 1
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1
            )
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate(
                [np.ones((1, max_len - tlen, act_dim)) * -1.0, a[-1]], axis=1
            )
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            rtg[-1] = (
                np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1)
                / scale
            )
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1
            )
            mask.append(
                np.concatenate(
                    [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1
                )
            )

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=act_dtype)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long
        )
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(dtype=torch.bool)

        return s, a, r, rtg, timesteps, mask

    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum


@dataclass
class DecisionTransformerTrainerConfig:
    batch_size: int = 64
    learning_rate: float = 0.0001
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999
    max_iters: int = 100_000
    warmup_iters: int = 10_000
    device: str = "cuda"
    decay_lr: bool = False
    lr_decay_iters: int = 100_000
    min_lr: float = 1e-5
    pct_traj: float = 1.0
    reward_scale: float = 1000.0
    grad_clip: float = 0.25
    gradient_accumulation_steps: int = 1
    always_save_checkpoint: bool = False
    out_dir: str = "out"
    eval_only: bool = False
    eval_interval: int = 1000
    eval_iters: int = 100
    log_interval: int = 100


class DecisionTransformerTrainer:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config

    def train(self):
        # Get stats from the dataset
        self.dataset_stats_ = calculate_dataset_stats(self.dataset)

        # TODO: switch whether or not use priority sampling
        num_timesteps = sum(self.dataset_stats_.traj_lens)
        num_timesteps = max(int(self.config.pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(self.dataset_stats_.returns)  # lowest to highest
        num_trajectories = 1
        timesteps = self.dataset_stats_.traj_lens[sorted_inds[-1]]
        ind = len(self.dataset) - 2
        while (
            ind >= 0
            and timesteps + self.dataset_stats_.traj_lens[sorted_inds[ind]]
            <= num_timesteps
        ):
            timesteps += self.dataset_stats_.traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]

        # used to reweight sampling so we sample according to timesteps instead of trajectories
        p_sample = self.dataset_stats_.traj_lens[sorted_inds] / sum(
            self.dataset_stats_.traj_lens[sorted_inds]
        )

        # TODO: Support resuming from checkpoint
        # optimizer
        optimizer = self.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            (self.config.beta1, self.config.beta2),
            self.config.device,
        )

        collator = DecisionTransformerDataCollator(
            state_mean=self.dataset_stats_.state_mean,
            state_std=self.dataset_stats_.state_std,
            K=self.model.config.K,
            max_ep_len=self.model.config.max_ep_len,
            reward_scale=self.config.reward_scale,
            act_discrete=self.model.config.act_discrete,
        )
        # WeightedRandomSampler
        n_samples = (
            self.config.max_iters
            * self.config.batch_size
            * self.config.gradient_accumulation_steps
        )
        n_samples *= 2  # TODO: Dirty hack to make it enough for the whole dataset with evaluation
        sampler = WeightedRandomSampler(
            weights=p_sample,
            num_samples=n_samples,  # Sample as many as the dataset size per epoch
            replacement=True,  # Allow replacement for sampling
        )
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            collate_fn=collator,
            sampler=sampler,
            num_workers=2,  # Use multiple workers
            # pin_memory=True  # Pin memory for faster transfer to GPU
        )
        dataloader_iter = iter(dataloader)

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        iter_num = 0
        best_val_loss = 1e9

        self.model.to(self.config.device)

        # training loop
        states, actions, rewards, rtgs, tsteps, mask = next(
            dataloader_iter
        )  # fetch the very first batch
        t0 = time.time()
        local_iter_num = 0  # number of iterations in the lifetime of this process
        raw_model = self.model  # unwrap DDP container if needed
        running_mfu = -1.0
        while True:
            # determine and set the learning rate for this iteration
            lr = (
                self.get_lr(iter_num)
                if self.config.decay_lr
                else self.config.learning_rate
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % self.config.eval_interval == 0:
                losses = self.estimate_loss(dataloader_iter)
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                # if losses["val"] < best_val_loss or self.config.always_save_checkpoint:
                #     best_val_loss = losses["val"]
                #     if iter_num > 0:
                #         checkpoint = {
                #             "model": raw_model.state_dict(),
                #             "optimizer": optimizer.state_dict(),
                #             "model_args": self.model.args,
                #             "iter_num": iter_num,
                #             "best_val_loss": best_val_loss,
                #             "trainer_args": self.args,
                #         }
                #         print(f"saving checkpoint to {self.out_dir}")
                #         Path(self.out_dir).mkdir(parents=True, exist_ok=True)
                #         torch.save(checkpoint, os.path.join(self.out_dir, "ckpt.pt"))
            if iter_num == 0 and self.config.eval_only:
                break

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            for micro_step in range(self.config.gradient_accumulation_steps):
                # move to device
                states, actions, rewards, rtgs, tsteps, mask = (
                    states.to(self.config.device),
                    actions.to(self.config.device),
                    rewards.to(self.config.device),
                    rtgs.to(self.config.device),
                    tsteps.to(self.config.device),
                    mask.to(self.config.device),
                )

                logits, loss = self.model(
                    states, actions, rtgs, tsteps, mask, targets=actions
                )
                loss = (
                    loss / self.config.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # I guess this makes sense when ddp is used, but it was removed
                states, actions, rewards, rtgs, tsteps, mask = next(dataloader_iter)
                # backward pass, with gradient scaling if training in fp16
                loss.backward()
            # clip the gradient
            if self.config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            # step the optimizer and scaler if training in fp16
            optimizer.step()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % self.config.log_interval == 0:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * self.config.gradient_accumulation_steps
                if local_iter_num >= 5:  # let the training loop settle a bit
                    mfu = self.estimate_mfu(
                        self.config.batch_size
                        * self.config.gradient_accumulation_steps,
                        dt,
                    )
                    running_mfu = (
                        mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                    )
                print(
                    f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
                )
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > self.config.max_iters:
                break

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self, dataloader_iter):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                states, actions, rewards, rtgs, tsteps, mask = next(dataloader_iter)
                states, actions, rewards, rtgs, tsteps, mask = (
                    states.to(self.config.device),
                    actions.to(self.config.device),
                    rewards.to(self.config.device),
                    rtgs.to(self.config.device),
                    tsteps.to(self.config.device),
                    mask.to(self.config.device),
                )
                logits, loss = self.model(
                    states, actions, rtgs, tsteps, mask, targets=actions
                )
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.model.get_num_params()
        cfg = self.model.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.max_ep_len
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


class TrajectoryDataset(Dataset):
    def __init__(self, trajectories):
        self.trajectories = trajectories
        self._calculate_stats()

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]

    def _calculate_stats(self):
        states, traj_lens, returns = [], [], []
        for path in self.trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        self.traj_lens_, self.returns_ = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.vstack(states)
        self.state_mean_, self.state_std_ = (
            np.mean(states, axis=0),
            np.std(states, axis=0) + 1e-6,
        )


@dataclass
class DatasetStats:
    state_mean: float
    state_std: float
    traj_lens: npt.NDArray
    returns: npt.NDArray


def calculate_dataset_stats(dataset: MinariDataset):
    states, traj_lens, returns = [], [], []
    for path in dataset:
        states.append(path.observations)
        traj_lens.append(len(path.observations))
        returns.append(path.rewards.sum())

    traj_lens_, returns_ = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.vstack(states)
    state_mean_, state_std_ = (
        np.mean(states, axis=0),
        np.std(states, axis=0) + 1e-6,
    )

    return DatasetStats(state_mean_, state_std_, traj_lens_, returns_)
