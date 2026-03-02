# net/tag_layer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _row_norm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return A / (A.sum(-1, keepdim=True) + eps)


def _sym_norm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # A: (..., M, M)
    d = A.sum(-1)  # (..., M)
    d_inv_sqrt = torch.rsqrt(d + eps)
    D = torch.diag_embed(d_inv_sqrt)  # (..., M, M)
    return D @ A @ D


def _build_knn_adj(pos: torch.Tensor, k: int, self_loop: bool = True) -> torch.Tensor:
    """
    pos: (N, M, 3) -> A: (N, M, M) (0/1 edges, +I if self_loop)
    """
    N, M, _ = pos.shape
    d = torch.cdist(pos, pos, p=2)  # (N,M,M)

    if not self_loop:
        d = d + torch.eye(M, device=pos.device).unsqueeze(0) * 1e6

    k_eff = max(1, min(int(k), M))
    idx = d.topk(k_eff, largest=False, dim=-1).indices  # (N, M, k)

    A = torch.zeros((N, M, M), device=pos.device, dtype=pos.dtype)
    A.scatter_(-1, idx, 1.0)

    if self_loop:
        A = A + torch.eye(M, device=pos.device, dtype=pos.dtype).unsqueeze(0)
    return A


def _build_star_from_center(center: torch.Tensor, M: int, self_loop: bool = True) -> torch.Tensor:
    """
    center: (N,) long
    return A_star: (N,M,M) where row/col center connected to all.
    """
    N = center.numel()
    device = center.device
    A = torch.zeros((N, M, M), device=device, dtype=torch.float32)

    n_idx = torch.arange(N, device=device)
    ar = torch.arange(M, device=device)

    # row: center -> all
    A[n_idx, center, :] = 1.0
    # col: all -> center
    A[n_idx[:, None], ar[None, :], center[:, None]] = 1.0

    if self_loop:
        A = A + torch.eye(M, device=device, dtype=torch.float32).unsqueeze(0)
    return A


def _build_ball_star_soft(ball_score: torch.Tensor, tau_center: float = 0.35, self_loop: bool = True) -> torch.Tensor:
    """
    ball_score: (N,M) (higher means more likely ball handler)
    returns A_ball_soft: (N,M,M) = sum_m p(m) * A_star(center=m)
    This avoids hard argmax switching.
    """
    N, M = ball_score.shape
    device = ball_score.device

    tau = max(1e-6, float(tau_center))
    p = F.softmax(ball_score / tau, dim=1)  # (N,M)

    # build all stars in a vectorized way:
    # A_star(n, i, j) = 1 if (i==center) or (j==center), then weighted by p(center)
    # => A_soft(n,i,j) = p(n,i) + p(n,j) - (i==j ? p(n,i) : 0)  (double counts diag; handle later)
    # Easier & safe: construct with broadcasting:
    pi = p.unsqueeze(2)          # (N,M,1)
    pj = p.unsqueeze(1)          # (N,1,M)
    A = pi + pj                  # (N,M,M)
    # If self_loop, we want diag >=1 anyway; if not self_loop, remove diagonal contribution
    if not self_loop:
        A = A - torch.diag_embed(p)  # remove diag
    else:
        # make sure identity exists strongly (same behavior as knn + I)
        A = A + torch.eye(M, device=device, dtype=A.dtype).unsqueeze(0)

    return A


@dataclass
class SoftSelectorConfig:
    enable: bool = False
    k_list: List[int] = None
    selector: str = "global"      # global | per_sample | per_class_prior
    num_class: int = 20
    warmup_epochs: int = 12
    gate_init: float = 0.0
    gate_max: float = 1.0
    temperature: float = 1.0


class TAGLayer(nn.Module):
    """
    TAG(x): (N,C,T,V,M) -> (N,C,T,V,M)

    - kNN MoE: alpha_k mixes A_knn(k)
    - Ball graph: A_ball_soft (soft center, no argmax switching)
    - Final adjacency: A = w_ball * A_ball_soft + (1-w_ball) * A_knn_mix
    - gate + warmup ensures TAG starts ~0 influence
    """

    def __init__(
        self,
        mode: str = "knn",          # "knn" or "ball"
        k: int = 4,
        lambda_fuse: float = 0.10,
        learnable_lambda: bool = True,
        self_loop: bool = True,
        norm: str = "sym",          # row / sym
        detach_adj: bool = True,
        use_ball: bool = True,
        fallback: str = "knn",
        tau: float = 0.35,          # used as ball-center temperature tau_center
        ramp_epochs: int = 12,
        alpha_selfloop: float = 0.5,  # reserved
        soft_selector: Optional[Dict] = None,

        # --- new: ball + knn mixing control ---
        ball_mix: bool = True,          # if mode=="ball", whether to mix with knn
        ball_weight: float = -1.0,      # if >=0 => fixed w_ball; if <0 => adaptive from score confidence
        ball_tau_center: float = -1.0,  # if >0 override tau
    ):
        super().__init__()
        self.mode = str(mode)
        self.k = int(k)
        self.self_loop = bool(self_loop)
        self.norm = str(norm)
        self.detach_adj = bool(detach_adj)
        self.use_ball = bool(use_ball)
        self.fallback = str(fallback)

        self.ball_mix = bool(ball_mix)
        self.ball_weight = float(ball_weight)
        self.tau_center = float(ball_tau_center) if float(ball_tau_center) > 0 else float(tau)

        # lambda
        if learnable_lambda:
            self.lambda_fuse = nn.Parameter(torch.tensor(float(lambda_fuse)))
        else:
            self.register_buffer("lambda_fuse", torch.tensor(float(lambda_fuse)))

        # epoch state
        self._epoch = 0

        # gate + warmup
        self.tag_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid gate
        self.warmup_epochs = int(ramp_epochs)
        self.gate_max = 1.0

        # soft selector
        ss = soft_selector or {}
        self.soft = SoftSelectorConfig(
            enable=bool(ss.get("enable", False)),
            k_list=[int(x) for x in ss.get("k_list", [])] if ss.get("k_list", None) is not None else None,
            selector=str(ss.get("selector", "global")),
            num_class=int(ss.get("num_class", 20)),
            warmup_epochs=int(ss.get("warmup_epochs", ramp_epochs)),
            gate_init=float(ss.get("gate_init", 0.0)),
            gate_max=float(ss.get("gate_max", 1.0)),
            temperature=float(ss.get("temperature", 1.0)),
        )

        if self.soft.enable:
            if not self.soft.k_list:
                self.soft.k_list = [2, 3, 4, 5]
            self.k_list = list(self.soft.k_list)
            self.warmup_epochs = self.soft.warmup_epochs
            self.gate_max = self.soft.gate_max
            with torch.no_grad():
                self.tag_gate.fill_(self.soft.gate_init)

            # selector head
            if self.soft.selector == "global":
                self.alpha_logits = nn.Parameter(torch.zeros(len(self.k_list)))
            elif self.soft.selector == "per_sample":
                self.alpha_mlp = nn.Sequential(
                    nn.Linear(8, 32),
                    nn.ReLU(inplace=True),
                    nn.Linear(32, len(self.k_list)),
                )
                nn.init.zeros_(self.alpha_mlp[-1].weight)
                nn.init.zeros_(self.alpha_mlp[-1].bias)
            elif self.soft.selector == "per_class_prior":
                self.register_buffer(
                    "class_prior",
                    torch.ones(self.soft.num_class, len(self.k_list)) / float(len(self.k_list)),
                )
            else:
                raise ValueError(f"Unknown soft selector: {self.soft.selector}")
        else:
            self.k_list = [self.k]

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def set_class_prior(self, prior: torch.Tensor):
        """
        prior: (num_class, K), rows sum to 1
        """
        if not (self.soft.enable and self.soft.selector == "per_class_prior"):
            return
        assert prior.ndim == 2 and prior.shape[1] == len(self.k_list)
        self.class_prior = prior.to(self.class_prior.device)

    def _ramp(self) -> float:
        if self.warmup_epochs <= 0:
            return 1.0
        e = float(self._epoch)
        return max(0.0, min(1.0, e / float(self.warmup_epochs)))

    def _normA(self, A: torch.Tensor) -> torch.Tensor:
        if self.norm == "row":
            return _row_norm(A)
        return _sym_norm(A)

    @staticmethod
    def _get_pos(x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,T,V,M) xyz in 0:3
        pos: (N,M,3) from hip-root mean over T
        """
        hip_l, hip_r = 11, 12
        root = 0.5 * (x[:, 0:3, :, hip_l, :] + x[:, 0:3, :, hip_r, :])  # (N,3,T,M)
        pos = root.mean(dim=2).permute(0, 2, 1).contiguous()            # (N,M,3)
        return pos

    @staticmethod
    def _ball_score(x: torch.Tensor) -> torch.Tensor:
        """
        x[:,3] is ball channel => score (N,M)
        """
        return x[:, 3, :, :, :].mean(dim=1).mean(dim=1)  # (N,M)

    def _alpha(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        return alpha: (N,K) or (1,K)
        """
        K = len(self.k_list)
        if not self.soft.enable:
            return torch.ones((1, K), device=x.device, dtype=x.dtype) / float(K)

        temp = max(1e-6, float(self.soft.temperature))

        if self.soft.selector == "global":
            a = F.softmax(self.alpha_logits / temp, dim=0).view(1, K)
            return a.to(device=x.device, dtype=x.dtype)

        if self.soft.selector == "per_sample":
            pos = self._get_pos(x)  # (N,M,3)
            dmean = torch.cdist(pos, pos).mean(dim=(1, 2))  # (N,)
            if self.use_ball and x.size(1) >= 4:
                s = self._ball_score(x)
            else:
                s = torch.zeros(pos.size(0), pos.size(1), device=x.device)
            feat = torch.stack(
                [
                    dmean,
                    pos[..., 0].std(dim=1),
                    pos[..., 1].std(dim=1),
                    pos[..., 2].std(dim=1),
                    s.mean(dim=1),
                    s.std(dim=1),
                    s.max(dim=1).values,
                    s.min(dim=1).values,
                ],
                dim=1,
            )  # (N,8)
            logits = self.alpha_mlp(feat)  # (N,K)
            return F.softmax(logits / temp, dim=1)

        if self.soft.selector == "per_class_prior":
            assert label is not None, "per_class_prior needs labels in training"
            prior = self.class_prior[label]  # (N,K)
            prior = prior / (prior.sum(dim=1, keepdim=True) + 1e-6)
            return prior.to(device=x.device, dtype=x.dtype)

        raise RuntimeError("unreachable")

    def _ball_weight(self, ball_score: torch.Tensor) -> torch.Tensor:
        """
        return w_ball: (N,1) in [0,1]
        If self.ball_weight>=0 => fixed scalar.
        Else => adaptive using confidence of ball_score distribution.
        """
        N, M = ball_score.shape
        if self.ball_weight >= 0:
            w = torch.full((N, 1), float(self.ball_weight), device=ball_score.device, dtype=ball_score.dtype)
            return w.clamp(0.0, 1.0)

        # adaptive: based on peakiness of p(m)
        tau = max(1e-6, float(self.tau_center))
        p = F.softmax(ball_score / tau, dim=1)         # (N,M)
        p_max = p.max(dim=1).values.unsqueeze(1)       # (N,1)
        # map [1/M, 1] -> [0,1]
        w = (p_max - (1.0 / M)) / (1.0 - (1.0 / M) + 1e-6)
        return w.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (N,C,T,V,M)
        """
        N, C, T, V, M = x.shape
        pos = self._get_pos(x)  # (N,M,3)

        K = len(self.k_list)

        # --- build kNN stack always (for mixing / fallback / soft selector) ---
        A_knn_list = [_build_knn_adj(pos, k=k, self_loop=self.self_loop) for k in self.k_list]
        A_knn_stack = torch.stack(A_knn_list, dim=1)  # (N,K,M,M)

        # normalize knn stack
        A_knn_stack = self._normA(A_knn_stack.view(N * K, M, M)).view(N, K, M, M)
        if self.detach_adj:
            A_knn_stack = A_knn_stack.detach()

        # alpha for knn mix
        alpha = self._alpha(x, label=label)  # (1,K) or (N,K)
        if alpha.size(0) == 1:
            alpha = alpha.expand(N, -1)
        A_knn = torch.einsum("nk,nkij->nij", alpha, A_knn_stack)  # (N,M,M)

        # --- ball soft star (optional) ---
        use_ball_now = (self.use_ball and C >= 4)
        if self.mode == "ball" and use_ball_now:
            s = self._ball_score(x)  # (N,M)
            A_ball = _build_ball_star_soft(s, tau_center=self.tau_center, self_loop=self.self_loop)  # (N,M,M)
            A_ball = self._normA(A_ball)  # normalize too
            if self.detach_adj:
                A_ball = A_ball.detach()

            if self.ball_mix:
                w = self._ball_weight(s).unsqueeze(-1)  # (N,1,1)
                A = w * A_ball + (1.0 - w) * A_knn
            else:
                A = A_ball
        else:
            # knn only
            A = A_knn

        # --- message passing on root trajectory (stable) ---
        hip_l, hip_r = 11, 12
        root = 0.5 * (x[:, :, :, hip_l, :] + x[:, :, :, hip_r, :])      # (N,C,T,M)
        root = root.permute(0, 2, 3, 1).contiguous()                    # (N,T,M,C)
        agg = torch.einsum("nij,ntjc->ntic", A, root)                   # (N,T,M,C)

        lam = self.lambda_fuse
        root2 = root + lam * agg                                        # (N,T,M,C)
        root2 = root2.permute(0, 3, 1, 2).contiguous()                  # (N,C,T,M)

        # residual delta broadcast to all joints
        root_old = 0.5 * (x[:, :, :, hip_l, :] + x[:, :, :, hip_r, :])  # (N,C,T,M)
        delta = (root2 - root_old).unsqueeze(3)                         # (N,C,T,1,M)
        x2 = x + delta.expand(N, C, T, V, M)

        # gate + warmup
        gate = torch.sigmoid(self.tag_gate) * float(self.gate_max)
        ramp = float(self._ramp())
        x_out = x + (gate * ramp) * (x2 - x)

        return x_out
