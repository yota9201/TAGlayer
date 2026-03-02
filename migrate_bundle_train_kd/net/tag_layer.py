# net/tag_layer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict

import math

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



# ===== Team / Matchup graph helpers (SGA-INTERACT friendly) =====
# M is expected to be 6 (3v3). We infer two teams per-clip by enumerating all 3-3 partitions (10 total).
# This is robust to cross-clip track-ID permutations (M order can change across clips).

def _enumerate_3v3_partitions(M: int, device=None) -> torch.Tensor:
    """Return partitions as (P,3) long indices, with duplicates removed (P=10 when M=6)."""
    if M != 6:
        raise ValueError("3v3 partitions only implemented for M=6")
    import itertools
    parts = []
    for comb in itertools.combinations(range(M), 3):
        # remove symmetric duplicates by fixing that 0 must be in the first group
        if 0 not in comb:
            continue
        parts.append(comb)
    idx = torch.tensor(parts, dtype=torch.long, device=device)  # (10,3)
    return idx

def _infer_teams_by_partition(pos: torch.Tensor,
                             ball_score: torch.Tensor,
                             gamma_between: float = 0.5,
                             eta_ball: float = 0.5) -> torch.Tensor:
    """
    pos: (N,M,3) clip-level mean positions
    ball_score: (N,M) clip-level possession score (mean of ball01 in time)
    Return team_id: (N,M) long in {0,1} (two teams), inferred per sample.
    """
    N, M, _ = pos.shape
    device = pos.device
    parts = _enumerate_3v3_partitions(M, device=device)  # (P,3)
    P = parts.size(0)

    # pairwise distances (N,M,M)
    D = torch.cdist(pos, pos)  # euclidean

    # precompute masks for each partition: (P,M)
    maskA = torch.zeros((P, M), device=device, dtype=torch.bool)
    maskA.scatter_(1, parts, True)
    maskB = ~maskA

    # compute within sums and between means in a vectorized way
    # withinA: sum_{i<j in A} D_ij ; withinB similarly
    # We'll use mask outer products.
    mA2 = maskA.unsqueeze(2) & maskA.unsqueeze(1)  # (P,M,M)
    mB2 = maskB.unsqueeze(2) & maskB.unsqueeze(1)

    # remove diagonal
    eye = torch.eye(M, device=device, dtype=torch.bool).unsqueeze(0)
    mA2 = mA2 & ~eye
    mB2 = mB2 & ~eye

    # within sum: sum over i,j then /2 since symmetric counted twice
    withinA = (D.unsqueeze(1) * mA2.unsqueeze(0).to(D.dtype)).sum(dim=(2,3)) * 0.5  # (N,P)
    withinB = (D.unsqueeze(1) * mB2.unsqueeze(0).to(D.dtype)).sum(dim=(2,3)) * 0.5  # (N,P)
    within = withinA + withinB  # (N,P)

    # between mean: mean D over i in A, j in B
    mAB = maskA.unsqueeze(2) & maskB.unsqueeze(1)  # (P,M,M) rows in A cols in B
    between_sum = (D.unsqueeze(1) * mAB.unsqueeze(0).to(D.dtype)).sum(dim=(2,3))  # (N,P)
    between_cnt = mAB.sum(dim=(1,2)).clamp(min=1).to(D.dtype)  # (P,)
    between = between_sum / between_cnt.unsqueeze(0)  # (N,P)

    # ball split: encourage possession concentrated in one team
    bA = (ball_score.unsqueeze(1) * maskA.unsqueeze(0).to(ball_score.dtype)).sum(dim=2)  # (N,P)
    bB = (ball_score.unsqueeze(1) * maskB.unsqueeze(0).to(ball_score.dtype)).sum(dim=2)  # (N,P)
    ball_split = (bA - bB).abs()  # (N,P)

    score = within - float(gamma_between) * between - float(eta_ball) * ball_split  # (N,P)
    best = score.argmin(dim=1)  # (N,)

    best_maskA = maskA[best]  # (N,M)
    # team_id: 1 for maskA, 0 for maskB
    team_id = best_maskA.long()
    return team_id  # (N,M) in {0,1}

def _build_knn_within_mask(pos: torch.Tensor,
                           mask: torch.Tensor,
                           k: int,
                           self_loop: bool = True) -> torch.Tensor:
    """
    pos: (N,M,3), mask: (N,M) bool; only nodes with mask=True connect within that set.
    returns A: (N,M,M)
    """
    N, M, _ = pos.shape
    d = torch.cdist(pos, pos)  # (N,M,M)

    # valid pairs within the group
    m2 = mask.unsqueeze(2) & mask.unsqueeze(1)  # (N,M,M)
    big = torch.tensor(1e6, device=pos.device, dtype=pos.dtype)

    d = d.masked_fill(~m2, big)
    if not self_loop:
        d = d + torch.eye(M, device=pos.device, dtype=pos.dtype).unsqueeze(0) * big

    k_eff = max(1, min(int(k), M))
    idx = d.topk(k_eff, largest=False, dim=-1).indices  # (N,M,k)

    A = torch.zeros((N, M, M), device=pos.device, dtype=pos.dtype)
    A.scatter_(-1, idx, 1.0)

    # remove rows for nodes outside mask
    A = A * mask.unsqueeze(-1).to(A.dtype)

    if self_loop:
        A = A + torch.eye(M, device=pos.device, dtype=pos.dtype).unsqueeze(0)
    return A

def _build_matchup_adj(pos: torch.Tensor,
                       maskA: torch.Tensor,
                       maskB: torch.Tensor,
                       k_match: int = 1,
                       self_loop: bool = True,
                       symmetric: bool = True) -> torch.Tensor:
    """
    Build cross-team (matchup) adjacency between A and B by nearest neighbors.
    pos: (N,M,3), maskA/maskB: (N,M) bool disjoint
    returns A_match: (N,M,M)
    """
    N, M, _ = pos.shape
    d = torch.cdist(pos, pos)  # (N,M,M)
    big = torch.tensor(1e6, device=pos.device, dtype=pos.dtype)

    # A->B edges
    mAB = maskA.unsqueeze(2) & maskB.unsqueeze(1)  # rows in A, cols in B
    dAB = d.masked_fill(~mAB, big)
    k_eff = max(1, min(int(k_match), M))
    idxAB = dAB.topk(k_eff, largest=False, dim=-1).indices  # (N,M,k)

    A = torch.zeros((N, M, M), device=pos.device, dtype=pos.dtype)
    A.scatter_(-1, idxAB, 1.0)
    A = A * maskA.unsqueeze(-1).to(A.dtype)

    if symmetric:
        # B->A
        mBA = maskB.unsqueeze(2) & maskA.unsqueeze(1)
        dBA = d.masked_fill(~mBA, big)
        idxBA = dBA.topk(k_eff, largest=False, dim=-1).indices
        A2 = torch.zeros((N, M, M), device=pos.device, dtype=pos.dtype)
        A2.scatter_(-1, idxBA, 1.0)
        A2 = A2 * maskB.unsqueeze(-1).to(A2.dtype)
        A = A + A2

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

        # --- new: team / matchup graphs (per-clip team inference) ---
        team_graph: bool = False,          # enable team-partition + matchup graphs
        team_gamma_between: float = 0.5,   # encourages between-team separation in partition score
        team_eta_ball: float = 0.5,        # encourages ball possession concentrated in one team
        lambda_off: float = 0.35,          # weight for offense intra-team graph
        lambda_def: float = 0.20,          # weight for defense intra-team graph
        lambda_match: float = 0.35,        # weight for matchup (cross-team) graph
        k_def: int = 2,                    # fixed k for defense intra-team kNN
        k_match: int = 1,                  # nearest defenders per attacker (and vice versa if symmetric)
        matchup_symmetric: bool = True,    # make matchup edges bidirectional
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

        # team / matchup graphs
        self.team_graph = bool(team_graph)
        self.team_gamma_between = float(team_gamma_between)
        self.team_eta_ball = float(team_eta_ball)
        self.lambda_off = float(lambda_off)
        self.lambda_def = float(lambda_def)
        self.lambda_match = float(lambda_match)
        self.k_def = int(k_def)
        self.k_match = int(k_match)
        self.matchup_symmetric = bool(matchup_symmetric)

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

        # --- clip-level ball score (possession) ---
        use_ball_now = (self.use_ball and C >= 4)
        s_ball = self._ball_score(x) if use_ball_now else None  # (N,M) in [0,1]

        # --- build kNN stack (used for selector / fallback; also used when team_graph=False) ---
        A_knn_list = [_build_knn_adj(pos, k=k, self_loop=self.self_loop) for k in self.k_list]
        A_knn_stack = torch.stack(A_knn_list, dim=1)  # (N,K,M,M)
        A_knn_stack = self._normA(A_knn_stack.view(N * K, M, M)).view(N, K, M, M)
        if self.detach_adj:
            A_knn_stack = A_knn_stack.detach()

        # alpha for kNN MoE
        alpha = self._alpha(x, label=label)  # (1,K) or (N,K)
        if alpha.size(0) == 1:
            alpha = alpha.expand(N, -1)

        # ===== Option A: Team + Matchup graphs (recommended for SGA-INTERACT) =====
        if self.team_graph and (M == 6) and use_ball_now:
            # infer two teams per sample (robust to cross-clip track ID permutation)
            team_id = _infer_teams_by_partition(
                pos=pos,
                ball_score=s_ball,
                gamma_between=self.team_gamma_between,
                eta_ball=self.team_eta_ball,
            )  # (N,M) in {0,1}

            # offense team = the team with higher total ball possession in the clip
            b_team1 = (s_ball * team_id.to(s_ball.dtype)).sum(dim=1)               # (N,)
            b_team0 = (s_ball * (1 - team_id).to(s_ball.dtype)).sum(dim=1)         # (N,)
            off_is_team1 = (b_team1 >= b_team0)                                    # (N,)

            off_mask = torch.where(off_is_team1.unsqueeze(1), team_id.bool(), (~team_id.bool()))  # (N,M)
            def_mask = ~off_mask

            # --- ball-centric anchor graph: soft star (avoids argmax switching) ---
            A_ball = _build_ball_star_soft(s_ball, tau_center=self.tau_center, self_loop=self.self_loop)

            # --- offense intra-team MoE(k) ---
            A_off_list = [_build_knn_within_mask(pos, off_mask, k=k, self_loop=self.self_loop) for k in self.k_list]
            A_off_stack = torch.stack(A_off_list, dim=1)  # (N,K,M,M)
            A_off_stack = self._normA(A_off_stack.view(N * K, M, M)).view(N, K, M, M)
            if self.detach_adj:
                A_off_stack = A_off_stack.detach()
            A_off = torch.einsum("nk,nkij->nij", alpha, A_off_stack)  # (N,M,M)

            # --- defense intra-team (fixed small k, no selector) ---
            A_def = _build_knn_within_mask(pos, def_mask, k=self.k_def, self_loop=self.self_loop)
            A_def = self._normA(A_def)
            if self.detach_adj:
                A_def = A_def.detach()

            # --- matchup (cross-team) graph ---
            A_mat = _build_matchup_adj(
                pos=pos,
                maskA=off_mask,
                maskB=def_mask,
                k_match=self.k_match,
                self_loop=self.self_loop,
                symmetric=self.matchup_symmetric,
            )
            A_mat = self._normA(A_mat)
            if self.detach_adj:
                A_mat = A_mat.detach()

            # combine (ball anchor always on)
            A = A_ball + self.lambda_off * A_off + self.lambda_def * A_def + self.lambda_match * A_mat
            A = self._normA(A)

        else:
            # ===== Legacy: global kNN MoE + optional ball-mix =====
            A_knn = torch.einsum("nk,nkij->nij", alpha, A_knn_stack)  # (N,M,M)

            if self.mode == "ball" and use_ball_now:
                A_ball = _build_ball_star_soft(s_ball, tau_center=self.tau_center, self_loop=self.self_loop)

                if self.ball_mix:
                    # w_ball: fixed if ball_weight>=0, else based on confidence of possession distribution
                    if self.ball_weight >= 0.0:
                        w = torch.full((N, 1, 1), float(self.ball_weight), device=x.device, dtype=x.dtype)
                    else:
                        # confidence from entropy of p(ball_handler)
                        p = F.softmax(s_ball / max(1e-6, float(self.tau_center)), dim=1)  # (N,M)
                        ent = -(p * (p + 1e-6).log()).sum(dim=1, keepdim=True)             # (N,1)
                        ent = ent / math.log(float(M) + 1e-6)
                        conf = (1.0 - ent).clamp(0.0, 1.0)                                # (N,1)
                        w = conf.view(N, 1, 1)
                    A = w * A_ball + (1.0 - w) * A_knn
                else:
                    A = A_ball
            else:
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
