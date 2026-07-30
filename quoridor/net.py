"""Policy/value ResNet and its batched GPU evaluator.

Architecture follows AlphaZero: a shared convolutional trunk of residual blocks
over the 9x9 board, splitting into a policy head over the 140-action space and a
scalar value head in [-1, 1] from the side-to-move's perspective.

The board is small (9x9), so depth is cheap and width matters more than in Go.
Defaults (10 blocks x 128 channels) fit comfortably on a 16GB card with room for
large inference batches, which is what actually determines self-play throughput.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quoridor.encoding import NUM_PLANES
from quoridor.fastrules import NUM_ACTIONS


@dataclass
class NetConfig:
    channels: int = 128
    blocks: int = 10
    head_channels: int = 32
    value_hidden: int = 256
    planes: int = NUM_PLANES
    actions: int = NUM_ACTIONS


class ResidualBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class QuoridorNet(nn.Module):
    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg or NetConfig()
        c = self.cfg.channels

        self.stem = nn.Sequential(
            nn.Conv2d(self.cfg.planes, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(c) for _ in range(self.cfg.blocks)])

        hc = self.cfg.head_channels
        self.policy_conv = nn.Sequential(
            nn.Conv2d(c, hc, 1, bias=False), nn.BatchNorm2d(hc), nn.ReLU(inplace=True)
        )
        self.policy_fc = nn.Linear(hc * 81, self.cfg.actions)

        self.value_conv = nn.Sequential(
            nn.Conv2d(c, hc, 1, bias=False), nn.BatchNorm2d(hc), nn.ReLU(inplace=True)
        )
        self.value_fc = nn.Sequential(
            nn.Linear(hc * 81, self.cfg.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.cfg.value_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1.0 / math.sqrt(m.in_features))
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(policy_logits, value)``; value is tanh-squashed to [-1, 1]."""
        x = self.trunk(self.stem(x))
        p = self.policy_fc(self.policy_conv(x).flatten(1))
        v = torch.tanh(self.value_fc(self.value_conv(x).flatten(1))).squeeze(-1)
        return p, v

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------- checkpoints


def save_checkpoint(path: str | Path, net: QuoridorNet, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"cfg": asdict(net.cfg), "state_dict": net.state_dict(), "meta": meta or {}},
        path,
    )


def load_checkpoint(
    path: str | Path, device: str = "cuda"
) -> tuple[QuoridorNet, dict]:
    blob = torch.load(path, map_location=device, weights_only=False)
    net = QuoridorNet(NetConfig(**blob["cfg"])).to(device)
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return net, blob.get("meta", {})


# ----------------------------------------------------------------- inference


class NetEvaluator:
    """Batched inference wrapper used by MCTS.

    Masks illegal actions *before* the softmax rather than zeroing probabilities
    afterwards, so the remaining priors keep their relative weighting instead of
    being distorted by probability mass assigned to impossible moves.

    ``graph_batches`` captures the forward pass as a CUDA graph for those exact
    batch sizes. At batch 1 -- the interactive-play case, where one game means
    one position per simulation -- eager execution is dominated by Python and
    kernel-launch overhead, and a graph replay of the identical kernels is
    several times faster. The network, weights, and outputs are unchanged; only
    the launch mechanism differs. Other batch sizes fall through to eager.
    """

    def __init__(
        self,
        net: QuoridorNet,
        device: str = "cuda",
        use_amp: bool = True,
        channels_last: bool = True,
        graph_batches: tuple[int, ...] = (),
    ):
        self.net = net.to(device).eval()
        self.device = device
        self.use_amp = use_amp and device.startswith("cuda")
        self.channels_last = channels_last and device.startswith("cuda")
        if self.channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)
        self._graphs: dict[int, tuple] = {}
        if device.startswith("cuda"):
            for b in graph_batches:
                self._capture(b)

    def _forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            return self.net(x)

    def _capture(self, batch: int) -> None:
        """Record the forward pass for a fixed batch size as a CUDA graph."""
        static_in = torch.zeros(
            batch, self.net.cfg.planes, 9, 9, device=self.device
        )
        if self.channels_last:
            static_in = static_in.contiguous(memory_format=torch.channels_last)

        # Standard capture recipe: warm up on a side stream first so cuDNN
        # autotuning and lazy init don't end up inside the recording.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            for _ in range(3):
                self._forward(static_in)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            static_logits, static_value = self._forward(static_in)
        self._graphs[batch] = (graph, static_in, static_logits, static_value)

    @torch.inference_mode()
    def evaluate(
        self, planes: np.ndarray, legal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``planes``: float32[B,8,9,9], ``legal``: uint8[B,140].

        Returns ``(policy[B,140] normalised over legal moves, value[B])``.
        """
        x = torch.from_numpy(planes).to(self.device, non_blocking=True)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        entry = self._graphs.get(x.shape[0])
        if entry is not None:
            graph, static_in, static_logits, static_value = entry
            static_in.copy_(x)
            graph.replay()
            logits, value = static_logits, static_value
        else:
            logits, value = self._forward(x)

        logits = logits.float()
        mask = torch.from_numpy(legal).to(self.device).bool()
        logits = logits.masked_fill(~mask, float("-inf"))
        policy = torch.softmax(logits, dim=1)
        # A position with no legal moves cannot arise, but guard against NaNs
        # poisoning the search tree if one ever did.
        policy = torch.nan_to_num(policy, nan=0.0)
        return policy.cpu().numpy(), value.float().cpu().numpy()


class UniformEvaluator:
    """Prior-free evaluator: uniform policy, zero value.

    Turns the neural MCTS into plain PUCT with no knowledge, which is the
    control condition for checking that search itself is wired up correctly.
    """

    def evaluate(
        self, planes: np.ndarray, legal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        counts = legal.sum(axis=1, keepdims=True).clip(min=1)
        policy = (legal.astype(np.float32) / counts).astype(np.float32)
        value = np.zeros(planes.shape[0], dtype=np.float32)
        return policy, value
