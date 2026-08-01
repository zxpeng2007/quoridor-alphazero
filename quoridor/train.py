"""Training step: AlphaZero policy + value loss.

Policy loss is soft-target cross-entropy against the MCTS visit distribution,
computed over *legal* actions only so it matches how the policy is consumed at
inference. Value loss is MSE against the game outcome.

bfloat16 autocast is used without a gradient scaler -- bf16 has the same exponent
range as fp32, so the underflow that makes fp16 need scaling does not arise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from quoridor.net import QuoridorNet


@dataclass
class TrainConfig:
    batch_size: int = 1024
    lr: float = 2e-3
    min_lr: float = 1e-4
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    grad_clip: float = 5.0
    use_amp: bool = True
    warmup_steps: int = 200


@dataclass
class TrainStats:
    steps: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    policy_acc: float = 0.0
    value_mae: float = 0.0
    grad_norm: float = 0.0

    def __str__(self) -> str:
        return (
            f"loss {self.total_loss:.4f} "
            f"(policy {self.policy_loss:.4f}, value {self.value_loss:.4f})  "
            f"top1 {100 * self.policy_acc:.1f}%  "
            f"value MAE {self.value_mae:.3f}  "
            f"|g| {self.grad_norm:.2f}"
        )


class Trainer:
    def __init__(
        self,
        net: QuoridorNet,
        cfg: TrainConfig | None = None,
        device: str = "cuda",
        total_steps: int | None = None,
    ):
        self.cfg = cfg or TrainConfig()
        self.device = device
        self.net = net.to(device)
        if device.startswith("cuda"):
            self.net = self.net.to(memory_format=torch.channels_last)
        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        self.total_steps = total_steps
        self.step_count = 0
        self.use_amp = self.cfg.use_amp and device.startswith("cuda")

    def _lr_at(self, step: int) -> float:
        cfg = self.cfg
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        if self.total_steps is None:
            return cfg.lr
        progress = (step - cfg.warmup_steps) / max(
            self.total_steps - cfg.warmup_steps, 1
        )
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + np.cos(np.pi * progress))
        return cfg.min_lr + (cfg.lr - cfg.min_lr) * cos

    def step(
        self,
        planes: np.ndarray,
        policy_target: np.ndarray,
        value_target: np.ndarray,
        legal: np.ndarray,
    ) -> TrainStats:
        self.net.train()
        lr = self._lr_at(self.step_count)
        for group in self.opt.param_groups:
            group["lr"] = lr

        x = torch.from_numpy(planes).to(self.device, non_blocking=True)
        if self.device.startswith("cuda"):
            x = x.contiguous(memory_format=torch.channels_last)
        pi = torch.from_numpy(policy_target).to(self.device, non_blocking=True)
        z = torch.from_numpy(value_target).to(self.device, non_blocking=True)
        mask = torch.from_numpy(legal).to(self.device, non_blocking=True).bool()

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            logits, value = self.net(x)

        logits = logits.float().masked_fill(~mask, float("-inf"))
        logp = F.log_softmax(logits, dim=1)
        # Zero-target entries contribute nothing; guard the -inf * 0 = nan case.
        policy_loss = -(pi * torch.nan_to_num(logp, neginf=0.0)).sum(dim=1).mean()
        value_loss = F.mse_loss(value.float(), z)
        loss = policy_loss + self.cfg.value_weight * value_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.cfg.grad_clip
        )
        self.opt.step()
        self.step_count += 1

        with torch.no_grad():
            acc = (logits.argmax(1) == pi.argmax(1)).float().mean().item()
            mae = (value.float() - z).abs().mean().item()

        return TrainStats(
            steps=self.step_count,
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            total_loss=loss.item(),
            policy_acc=acc,
            value_mae=mae,
            grad_norm=float(grad_norm),
        )

    @torch.no_grad()
    def value_mae_on(self, states: np.ndarray, targets: np.ndarray,
                     batch: int = 512) -> float:
        """Mean absolute value error on a held-out set, no search involved.

        The promotion and Elo gates play their games at 200 simulations, where a
        branch holding 0.6% of the prior receives about one visit. A value repair
        confined to such branches is therefore invisible to them -- it could be a
        real improvement and still be gated out. This measures the value head
        directly against deep-search labels, which is the thing being repaired.
        """
        from quoridor.encoding import ROT180_ACTION
        from quoridor.mcts import _encode_batch
        from quoridor.fastrules import NUM_ACTIONS, SCRATCH_SIZE, legal_mask

        n = len(states)
        if n == 0:
            return float("nan")
        rot = ROT180_ACTION.astype(np.int32)
        was_training = self.net.training
        self.net.eval()
        total = 0.0
        for s in range(0, n, batch):
            chunk = np.ascontiguousarray(states[s:s + batch])
            b = len(chunk)
            legal = np.zeros((b, NUM_ACTIONS), dtype=np.uint8)
            scratch = np.zeros((b, SCRATCH_SIZE), dtype=np.int32)
            for i in range(b):
                legal_mask(chunk[i], legal[i], scratch[i])
            planes = np.zeros((b, 8, 9, 9), dtype=np.float32)
            legal_c = np.zeros((b, NUM_ACTIONS), dtype=np.uint8)
            flipped = np.zeros(b, dtype=np.uint8)
            d0 = np.zeros((b, 81), dtype=np.int32)
            d1 = np.zeros((b, 81), dtype=np.int32)
            canon = np.zeros((b, chunk.shape[1]), dtype=np.uint8)
            _encode_batch(chunk, planes, legal, legal_c, flipped,
                          scratch, d0, d1, canon, rot)
            x = torch.from_numpy(planes).to(self.device)
            if self.device.startswith("cuda"):
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
                _, v = self.net(x)
            z = torch.from_numpy(
                np.ascontiguousarray(targets[s:s + batch])).to(self.device)
            total += (v.float().reshape(-1) - z).abs().sum().item()
        if was_training:
            self.net.train()
        return total / n

    def train_on_buffer(self, buffer, n_steps: int, log_every: int = 0) -> TrainStats:
        """Run ``n_steps`` optimiser steps, returning the mean of the last decile."""
        recent: list[TrainStats] = []
        keep = max(n_steps // 10, 1)
        for i in range(n_steps):
            planes, pi, z, legal = buffer.sample(self.cfg.batch_size)
            stats = self.step(planes, pi, z, legal)
            recent.append(stats)
            if len(recent) > keep:
                recent.pop(0)
            if log_every and (i + 1) % log_every == 0:
                print(f"    step {i + 1}/{n_steps}  {stats}", flush=True)
        return TrainStats(
            steps=self.step_count,
            policy_loss=float(np.mean([s.policy_loss for s in recent])),
            value_loss=float(np.mean([s.value_loss for s in recent])),
            total_loss=float(np.mean([s.total_loss for s in recent])),
            policy_acc=float(np.mean([s.policy_acc for s in recent])),
            value_mae=float(np.mean([s.value_mae for s in recent])),
            grad_norm=float(np.mean([s.grad_norm for s in recent])),
        )
