"""
Full definition of a Decision Transformer Model, all of it in this single file.
Based on Andrej Karpathy's nanoGPT implementation of OpenAI's GPT-2.

References:
1) nanoGPT implementation of OpenAI's GPT-2: https://github.com/karpathy/nanoGPT/blob/master/model.py
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, dropout, bias, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Consider installing Flash Attention for faster training"
            )

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x, attn_mask=None):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # Build a mask to prevent attention to future tokens and padding tokens
        if attn_mask is not None:
            attn_mask = attn_mask.view(B, 1, 1, T)
            attn_mask = attn_mask & self.bias[:, :, :T, :T]
        else:
            attn_mask = self.bias[:, :, :T, :T]

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            float_mask = torch.zeros(B, 1, 1, T).to(
                device=attn_mask.device
            )  # Start with a tensor of zeros
            float_mask = float_mask.masked_fill(
                ~attn_mask, -10000
            )  # Fill masked positions with -10000

            dropout_p = self.dropout if self.training else 0
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=float_mask, dropout_p=dropout_p
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

            # Causal mask (for future tokens and padding tokens)
            # -10000.0 used instead of -inf to prevent nans when all values are masked
            # due to padding masking and causal masking
            att = att.masked_fill(~attn_mask, -10000.0)

            # Apply softmax to get attention probabilities
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Attention output
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, n_embd, n_head, dropout, bias, block_size):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, bias, block_size)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class DecisionTransformerConfig:
    n_layer: int = (3,)
    n_head: int = (1,)
    n_embd: int = (128,)
    dropout: float = (0.1,)
    bias: bool = (False,)
    K: int = (20,)
    max_ep_len: int = (1000,)
    state_dim: int = (17,)
    act_dim: int = (6,)
    act_discrete: bool = (False,)
    act_vocab_size: int = (1,)
    act_tanh: bool = (False,)
    tanh_embeddings: bool = (False,)


class DecisionTransformer(nn.Module):
    """This is basically a GPT-2 model with a few tweaks for Decision Transformer"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        print(f"Config: {config}")
        block_size = config.K * 3  # each block is composed of 3 tokens: R, s, a
        self.transformer = nn.ModuleDict(
            dict(
                te=nn.Embedding(config.max_ep_len, config.n_embd),
                re=nn.Linear(1, config.n_embd),
                se=nn.Linear(config.state_dim, config.n_embd),
                ae=(
                    nn.Embedding(config.act_vocab_size, config.n_embd)
                    if config.act_discrete
                    else nn.Linear(config.act_dim, config.n_embd)
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [
                        Block(
                            config.n_embd,
                            config.n_head,
                            config.dropout,
                            config.bias,
                            block_size,
                        )
                        for _ in range(config.n_layer)
                    ]
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
                ln_e=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        # TODO: consider bias=False in the act_head as in the original GPT lm_head
        if config.act_discrete:
            self.act_head = nn.Linear(config.n_embd, config.act_vocab_size)
        else:
            self.act_head = nn.Linear(config.n_embd, config.act_dim)

        # TODO: Let's experiment later if we can use weight tying for DT
        # self.transformer.ae.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.te.weight.numel()
            n_params -= self.transformer.se.weight.numel()
            n_params -= self.transformer.ae.weight.numel()
            n_params -= self.transformer.re.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, states, actions, rtgs, tsteps, attn_mask=None, targets=None):
        device = states.device
        b, t = states.shape[0], states.shape[1]
        assert (
            t <= self.config.K
        ), f"Cannot forward sequence of length {t}, K is only {self.config.K}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        state_emb = self.transformer.se(
            states
        )  # state embeddings of shape (b, t, n_embd)
        action_emb = self.transformer.ae(
            actions.type(torch.long).squeeze(-1)
            if self.config.act_discrete
            else actions
        )  # action embeddings of shape (b, t, n_embd)
        rtg_emb = self.transformer.re(
            rtgs
        )  # return-to-go embeddings of shape (b, t, n_embd)
        tstep_emb = self.transformer.te(
            tsteps
        )  # time / position embeddings of shape (t, n_embd)

        if self.config.tanh_embeddings:
            state_emb = torch.tanh(state_emb)
            action_emb = torch.tanh(action_emb)
            rtg_emb = torch.tanh(rtg_emb)

        # time embeddings are treated similar to positional embeddings
        state_emb = state_emb + tstep_emb
        action_emb = action_emb + tstep_emb
        rtg_emb = rtg_emb + tstep_emb

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_emb = (
            torch.stack((rtg_emb, state_emb, action_emb), dim=1)
            .permute(0, 2, 1, 3)
            .reshape(b, 3 * t, self.config.n_embd)
        )
        # TODO: Check if this LayerNorm is needed (some implementations don't have it)
        stacked_emb = self.transformer.ln_e(stacked_emb)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attn_mask = (
            torch.stack((attn_mask, attn_mask, attn_mask), dim=1)
            .permute(0, 2, 1)
            .reshape(b, 3 * t)
        )

        x = self.transformer.drop(stacked_emb)
        for block in self.transformer.h:
            x = block(x, stacked_attn_mask)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.act_head(x)

            if self.config.act_tanh:
                logits = torch.tanh(logits)

            logits = logits[:, 1::3, :]  # only keep predictions from state_embeddings

            # On cpu there are useful asserts
            # logits = logits.to("cpu")
            # targets = targets.to("cpu")

            if self.config.act_discrete:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
                )
            else:
                act_dim = logits.shape[2]
                logits = logits.reshape(-1, act_dim)[attn_mask.reshape(-1) > 0]
                targets = targets.reshape(-1, act_dim)[attn_mask.reshape(-1) > 0]
                loss = F.mse_loss(logits, targets)
        else:
            # inference-time mini-optimization: only forward the act_head on the very last position
            logits = self.act_head(
                x[:, [-2], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


class MultiGoalDecisionTransformer(DecisionTransformer):
    def forward(self, states, actions, rtgs, tsteps, attn_mask=None, targets=None):
        super().forward(states, actions, rtgs, tsteps, attn_mask, targets)
