# net/tag_layer.py
# TAG: Team/Player-level Adjacency over the person dimension (M).
# 输入/输出: (N, C, T, V, M)

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

# -------------------- Normalizers --------------------

def _row_normalize(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # adj: (..., M, M)
    deg = adj.sum(-1, keepdim=True).clamp(min=eps)
    return adj / deg

def _sym_normalize(adj: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    deg = adj.sum(-1).clamp(min=eps)            # (..., M)
    d_inv_sqrt = deg.pow(-0.5)                  # (..., M)
    return d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)

# -------------------- TAG Layer --------------------

class TAGLayer(nn.Module):
    """
    TAG: 在 '球员维(M)' 做一次消息传播，输出与输入同形状 (N,C,T,V,M)。
    - mode='knn': 基于位置的 kNN 软权重图（高斯权重）
    - mode='ball': 基于球权的星型图；若当帧无人持球，按 fallback='knn' 回退
    约定：
      * 通道 1~3 为 xyz 坐标（必须）
      * 通道 4（若存在）为球权 mask（每关节=1 表示该帧该人持球）
    训练技巧：
      * λ（融合系数）可学习
      * 余弦 warm-up：前 ramp_epochs 将 λ 从 0 平滑升到设定值
      * 自环强化 alpha_selfloop，缓解过平滑
    """

    def __init__(
        self,
        mode: str = 'knn',                # 'knn' | 'ball'
        k: int = 2,                       # kNN 中的 k
        lambda_fuse: float = 0.05,        # 残差融合系数
        learnable_lambda: bool = True,    # λ 是否可学习
        self_loop: bool = True,           # 是否加入自环
        alpha_selfloop: float = 0.5,      # 自环权重（加入后再整体归一化）
        norm: str = 'sym',                # 'row' | 'sym' | 'none'
        detach_adj: bool = True,          # 计算邻接时是否从计算图分离
        use_ball: bool = True,            # ball 模式是否使用第4通道
        fallback: str = 'knn',            # ball 模式下无人持球的回退方式
        tau: float = 0.35,                # kNN 的高斯温度: w=exp(-d/tau)
        ramp_epochs: int = 10,            # λ 的 warm-up epoch 数
        eps: float = 1e-6
    ):
        super().__init__()
        assert mode in ('knn', 'ball')
        assert norm in ('row', 'sym', 'none')
        self.mode = mode
        self.k = int(k)
        self.self_loop = bool(self_loop)
        self.alpha_selfloop = float(alpha_selfloop)
        self.norm = norm
        self.detach_adj = bool(detach_adj)
        self.use_ball = bool(use_ball)
        self.fallback = fallback
        self.tau = float(tau)
        self.ramp_epochs = int(max(0, ramp_epochs))
        self.eps = float(eps)

        lam = torch.tensor(float(lambda_fuse))
        self.lambda_fuse = nn.Parameter(lam) if bool(learnable_lambda) else lam

        # 训练时外部可写入当前 epoch（余弦 warm-up 用）
        # 例如：if model.tag is not None: model.tag._epoch = epoch
        self._epoch: Optional[int] = None

    # ----------- Utils -----------

    @staticmethod
    def _presence_mask(coords_xyz: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        coords_xyz: (N, 3, T, V, M)
        返回 present: (N, T, M)，若该帧该人所有关节近似为 0，则视为缺席。
        """
        assert coords_xyz.dim() == 5, f"expect (N,3,T,V,M), got {tuple(coords_xyz.shape)}"
        # 先在通道维(C=3)求和 -> (N,T,V,M)，再在关节维(V)求和 -> (N,T,M)
        mag = coords_xyz.abs().sum(dim=1).sum(dim=2)   # (N,T,M)
        present = mag > eps
        return present

    @staticmethod
    def _centers_from_xyz(coords_xyz: torch.Tensor) -> torch.Tensor:
        """
        coords_xyz: (N, 3, T, V, M)
        return: centers (N, T, M, 3)
        """
        assert coords_xyz.dim() == 5, f"expect (N,3,T,V,M), got {tuple(coords_xyz.shape)}"
        # 平均关节维 V -> (N, 3, T, M)
        centers = coords_xyz.mean(dim=3, keepdim=False)
        # 变换到 (N, T, M, 3)
        centers = centers.permute(0, 2, 3, 1).contiguous()
        return centers

    # ----------- kNN (soft) adjacency -----------

    def _adj_knn(self, pos: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """
        pos: (N,T,M,3) ; present: (N,T,M)
        高斯加权 kNN：w = exp(-d / tau)，对称化 + 自环 + 归一化
        """
        N, T, M, _ = pos.shape
        pd = pos.detach() if self.detach_adj else pos

        # pairwise distance
        dist = torch.cdist(pd, pd, p=2)                    # (N,T,M,M)
        big = torch.full_like(dist, 1e6)

        # 屏蔽缺席行/列
        pres_row = present.unsqueeze(-1).float()           # (N,T,M,1)
        pres_col = present.unsqueeze(-2).float()           # (N,T,1,M)
        dist = torch.where((pres_row * pres_col) > 0, dist, big)

        eye = torch.eye(M, device=dist.device).view(1, 1, M, M)
        dist = dist + eye * 1e6                            # 排除自身

        k = min(self.k, max(0, M - 1))
        if k == 0:
            adj = torch.zeros_like(dist)
        else:
            # 索引 & 高斯权重
            _, idx = torch.topk(dist, k=k, dim=-1, largest=False)  # (N,T,M,k)
            gather_dist = torch.gather(dist, -1, idx)              # (N,T,M,k)
            w = torch.exp(-gather_dist / self.tau)                  # (N,T,M,k)
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)           # 每行归一

            # 回散射到 (N,T,M,M)
            adj = torch.zeros_like(dist)
            adj.scatter_(-1, idx, w)

            # 对称化（平均更柔和）
            adj = 0.5 * (adj + adj.transpose(-1, -2))

        # 自环强化 + 归一化
        if self.self_loop:
            adj = adj + self.alpha_selfloop * eye
            adj = adj / (adj.sum(-1, keepdim=True) + 1e-6)

        # 最终归一化风格
        if self.norm == 'sym':
            adj = _sym_normalize(adj)
        elif self.norm == 'row':
            adj = _row_normalize(adj)
        return adj

    # ----------- Ball (star) adjacency -----------

    def _adj_ball(self, x: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,T,V,M), present: (N,T,M)
        ball 模式：持球人 ↔ 所有人；无人持球时根据 fallback 构造（默认 knn）。
        """
        N, C, T, V, M = x.shape
        device = x.device

        if not self.use_ball or C < 4:
            # 无球权信息 → 退化到 kNN
            centers = self._centers_from_xyz(x[:, :3, ...])
            return self._adj_knn(centers, present)

        # (N,T,V,M) → (N,T,M) : 任一关节>0.5 视为持球
        ball_map = x[:, 3, :, :, :]                         # (N,T,V,M)
        ball_mask = (ball_map > 0.5).float().amax(dim=2)    # (N,T,M)
        no_ball = (ball_mask.sum(dim=-1) < 0.5)             # (N,T) True=无人持球

        # 默认：星型图，持球人与所有人连边（含自环，后面统一处理）
        adj = torch.zeros(N, T, M, M, device=device)
        if (~no_ball).any():
            handler_idx = ball_mask.argmax(dim=-1)          # (N,T)
            n_idx = torch.arange(N, device=device)[:, None, None]  # (N,1,1)
            t_idx = torch.arange(T, device=device)[None, :, None]  # (1,T,1)
            h_idx = handler_idx[:, :, None]                 # (N,T,1)

            # 先连满星型
            adj[n_idx, t_idx, h_idx, :] = 1.0
            adj[n_idx, t_idx, :, h_idx] = 1.0

        # 缺席的人清零其行/列
        row_mask = present.float()                           # (N,T,M)
        adj = adj * row_mask.unsqueeze(-1)                   # 行
        adj = adj * row_mask.unsqueeze(-2)                   # 列

        # 无人持球帧：fallback
        if no_ball.any():
            if self.fallback == 'knn':
                centers = self._centers_from_xyz(x[:, :3, ...])
                adj_knn = self._adj_knn(centers, present)   # (N,T,M,M)
                mask = no_ball.unsqueeze(-1).unsqueeze(-1)  # (N,T,1,1)
                adj = torch.where(mask, adj_knn, adj)
            else:
                # 其他策略可在此扩展；目前仅 knn
                pass

        # 自环强化 + 归一
        eye = torch.eye(M, device=device).view(1, 1, M, M)
        if self.self_loop:
            adj = adj + self.alpha_selfloop * eye
        adj = adj / (adj.sum(-1, keepdim=True) + 1e-6)

        if self.norm == 'sym':
            adj = _sym_normalize(adj)
        elif self.norm == 'row':
            adj = _row_normalize(adj)
        return adj

    # ----------- Forward -----------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,T,V,M)
        返回: (N,C,T,V,M)
        """
        assert x.dim() == 5, f"Expect (N,C,T,V,M), got {tuple(x.shape)}"
        assert x.is_floating_point(), "TAGLayer expects floating-point tensor"
        N, C, T, V, M = x.shape
        coords_xyz = x[:, :3, :, :, :]                       # (N,3,T,V,M)
        present = self._presence_mask(coords_xyz, self.eps)  # (N,T,M)

        # 1) 动态邻接
        if self.mode == 'knn':
            centers = self._centers_from_xyz(coords_xyz)     # (N,T,M,3)
            A = self._adj_knn(centers, present)              # (N,T,M,M)
        else:  # 'ball'
            A = self._adj_ball(x, present)                   # (N,T,M,M)

        # 2) 沿 M 维传播： y[n,t,v,c,i] = Σ_j A[n,t,i,j] * x[n,c,t,v,j]
        x_ntvcm = x.permute(0, 2, 3, 1, 4).contiguous()      # (N,T,V,C,M)
        y_ntvci = torch.einsum('ntij,ntvcj->ntvci', A, x_ntvcm)
        y = y_ntvci.permute(0, 3, 1, 2, 4).contiguous()      # (N,C,T,V,M)

        # 3) 残差融合 + 余弦 warm-up
        if isinstance(self.lambda_fuse, torch.Tensor):
            lam_raw = self.lambda_fuse
        else:
            lam_raw = torch.tensor(float(self.lambda_fuse), device=x.device)

        lam = lam_raw
        if self.training and self.ramp_epochs > 0:
            epoch = self._epoch
            if epoch is not None:
                # scale from 0 -> 1 across ramp_epochs
                s = max(0.0, min(1.0, float(epoch) / float(self.ramp_epochs)))
                # cosine ramp
                scale = 0.5 - 0.5 * torch.cos(torch.tensor(s * 3.14159265, device=x.device))
                lam = lam * scale

        out = x + lam * y
        return out
