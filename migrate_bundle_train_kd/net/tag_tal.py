import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.shift_dual_stgcn_backbone import STGCNBackbone
from net.tag_layer import TAGLayer


class TemporalHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        num_class: int = 20,
        num_point: int = 17,
        num_person: int = 6,
        graph_args: Optional[Dict] = None,
        backbone_cls: str = "net.st_gcn.Model",
        backbone_args: Optional[Dict] = None,
        backbone_weights: Optional[str] = None,
        freeze_backbone: bool = True,
        person_pool: str = "mean",
        temporal_hidden: int = 256,
        dropout: float = 0.1,
        tag: Optional[Dict] = None,
    ):
        super().__init__()
        self.num_class = int(num_class)
        self._epoch = 0

        tag_cfg = tag or {}
        self.tag = None
        if bool(tag_cfg.get("enable", False)):
            self.tag = TAGLayer(
                mode=str(tag_cfg.get("mode", "ball")),
                k=int(tag_cfg.get("k", 4)),
                lambda_fuse=float(tag_cfg.get("lambda_fuse", 0.1)),
                learnable_lambda=bool(tag_cfg.get("learnable_lambda", True)),
                self_loop=bool(tag_cfg.get("self_loop", True)),
                norm=str(tag_cfg.get("norm", "sym")),
                detach_adj=bool(tag_cfg.get("detach_adj", True)),
                use_ball=bool(tag_cfg.get("use_ball", in_channels >= 4)),
                fallback=str(tag_cfg.get("fallback", "knn")),
                tau=float(tag_cfg.get("tau", 0.35)),
                ramp_epochs=int(tag_cfg.get("ramp_epochs", 12)),
                alpha_selfloop=float(tag_cfg.get("alpha_selfloop", 0.5)),
                soft_selector=tag_cfg.get("soft_selector"),
                ball_mix=bool(tag_cfg.get("ball_mix", True)),
                ball_weight=float(tag_cfg.get("ball_weight", -1.0)),
                ball_tau_center=float(tag_cfg.get("ball_tau_center", -1.0)),
                team_graph=bool(tag_cfg.get("team_graph", False)),
                team_gamma_between=float(tag_cfg.get("team_gamma_between", 0.5)),
                team_eta_ball=float(tag_cfg.get("team_eta_ball", 0.5)),
                lambda_off=float(tag_cfg.get("lambda_off", 0.35)),
                lambda_def=float(tag_cfg.get("lambda_def", 0.20)),
                lambda_match=float(tag_cfg.get("lambda_match", 0.35)),
                k_def=int(tag_cfg.get("k_def", 2)),
                k_match=int(tag_cfg.get("k_match", 1)),
                matchup_symmetric=bool(tag_cfg.get("matchup_symmetric", True)),
            )

        ba = dict(backbone_args or {})
        ba.setdefault("in_channels", int(in_channels))
        ba.setdefault("num_class", int(num_class))
        ba.setdefault("edge_importance_weighting", True)
        ba.setdefault("graph_args", graph_args or {})
        ba.pop("graph", None)

        self.backbone = STGCNBackbone(
            backbone_cls=backbone_cls,
            backbone_args=ba,
            weights=backbone_weights,
            ignore_prefix=None,
            freeze=freeze_backbone,
            person_pool=person_pool,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 32, num_point, num_person)
            feat = self.backbone(dummy)
            feat_dim = int(feat.size(1))

        self.temporal_head = TemporalHead(feat_dim, int(temporal_hidden), float(dropout))
        self.class_head = nn.Conv1d(int(temporal_hidden), self.num_class, kernel_size=1)
        self.actionness_head = nn.Conv1d(int(temporal_hidden), 1, kernel_size=1)

        nn.init.normal_(self.class_head.weight, 0, math.sqrt(2.0 / max(1, self.num_class)))
        nn.init.zeros_(self.class_head.bias)
        nn.init.normal_(self.actionness_head.weight, 0, math.sqrt(2.0))
        nn.init.zeros_(self.actionness_head.bias)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        if self.tag is not None and hasattr(self.tag, "set_epoch"):
            self.tag.set_epoch(epoch)

    def get_param_groups(self, base_lr: float, backbone_lr_mult: float = 0.1):
        backbone_lr = float(base_lr) * float(backbone_lr_mult)
        head_params = []
        backbone_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            else:
                head_params.append(p)

        groups = []
        if head_params:
            groups.append({"params": head_params, "lr": float(base_lr)})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": float(backbone_lr)})
        return groups

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.tag is not None:
            x = self.tag(x)

        feat = self.backbone(x)
        feat = feat.mean(dim=-1)
        hidden = self.temporal_head(feat)

        cls_logits = self.class_head(hidden)
        actionness_logits = self.actionness_head(hidden).squeeze(1)
        scores = torch.sigmoid(cls_logits) * torch.sigmoid(actionness_logits).unsqueeze(1)

        return {
            "cls_logits": cls_logits,
            "actionness_logits": actionness_logits,
            "scores": scores,
        }
