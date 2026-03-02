from typing import Dict, Optional

import torch
import torch.nn as nn

from net.shift_dual_stgcn_backbone import STGCNBackbone
from net.tag_layer import TAGLayer


class Model(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_class: int,
        num_point: int,
        num_person: int,
        tag: Optional[Dict] = None,
        backbone_cls: str = "net.st_gcn.Model",
        backbone_args: Optional[Dict] = None,
        backbone_weights: Optional[str] = None,
        freeze_backbone: bool = False,
        person_pool: str = "mean",
    ):
        super().__init__()
        self.num_class = int(num_class)
        self.tag = None

        tag_cfg = tag or {}
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

        self.classifier = nn.Conv1d(feat_dim, self.num_class, kernel_size=1)

    def set_epoch(self, epoch: int):
        if self.tag is not None and hasattr(self.tag, "set_epoch"):
            self.tag.set_epoch(epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tag is not None:
            x = self.tag(x)
        feat = self.backbone(x)
        feat = feat.mean(dim=-1)
        logits = self.classifier(feat)
        return logits

