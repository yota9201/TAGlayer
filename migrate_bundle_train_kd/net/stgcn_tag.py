import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# 你项目里一般都有 torchlight.import_class；但为稳妥这里自己写一个
def import_obj(path: str):
    import importlib
    mod_path, dot, attr = path.rpartition(".")
    if dot == "":
        return importlib.import_module(path)
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def _build_knn_adj(pos: torch.Tensor, k: int, self_loop: bool = True) -> torch.Tensor:
    """
    pos: (N,M,3)
    return A: (N,M,M) row-stochastic (each row sums to 1)
    """
    N, M, _ = pos.shape
    # pairwise dist: (N,M,M)
    d = torch.cdist(pos, pos, p=2)  # (N,M,M)
    # exclude self for knn selection if no self_loop
    if not self_loop:
        d = d + torch.eye(M, device=pos.device).unsqueeze(0) * 1e6

    k_eff = max(1, min(k, M))
    idx = d.topk(k_eff, largest=False, dim=-1).indices  # (N,M,k)

    A = torch.zeros((N, M, M), device=pos.device, dtype=pos.dtype)
    A.scatter_(-1, idx, 1.0)

    if self_loop:
        A = A + torch.eye(M, device=pos.device, dtype=pos.dtype).unsqueeze(0)

    # row normalize
    A = A / (A.sum(-1, keepdim=True) + 1e-6)
    return A


class Model(nn.Module):
    """
    ST-GCN backbone (optionally frozen) + TAG head (delta logits).
    输出给 processor 的始终是 logits: (N, num_class)
    """

    def __init__(
        self,
        num_class: int,
        backbone_cls: str = "net.st_gcn.Model",
        backbone_args: Optional[Dict[str, Any]] = None,
        backbone_weights: Optional[str] = None,
        freeze_backbone: bool = True,
        person_pool: str = "mean",  # mean / max
        tag: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        self.num_class = int(num_class)

        backbone_args = backbone_args or {}
        tag = tag or {}

        # ---- backbone
        Backbone = import_obj(backbone_cls)
        self.backbone = Backbone(**backbone_args)

        # ---- load backbone weights
        if backbone_weights:
            sd = torch.load(backbone_weights, map_location="cpu")
            sd = sd.get("state_dict", sd)
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f"[STGCNBackbone] loaded {backbone_weights}: missing={len(missing)}, unexpected={len(unexpected)}")

        self.freeze_backbone = bool(freeze_backbone)
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ---- TAG hyperparams
        self.person_pool = str(person_pool).lower()
        self.k = int(tag.get("k", 4))
        self.lambda_fuse = float(tag.get("lambda_fuse", 0.10))
        self.self_loop = bool(tag.get("self_loop", True))

        # warmup / gate
        self.tag_warmup_epochs = int(tag.get("warmup_epochs", tag.get("ramp_epochs", 10)))
        self.tag_scale_max = float(tag.get("tag_scale_max", 1.0))

        # learnable gate param (sigmoid)
        self.tag_scale = nn.Parameter(torch.tensor(0.0))

        # epoch set by processor
        self.cur_epoch = 0

        # ---- probe feature dim Cf using a small dummy forward on CPU
        # ST-GCN 有 extract_feature，可以拿到 feat: (N,C,T,V,M)
        self.feat_dim = int(tag.get("feat_dim", 256))  # 默认 256（ST-GCN 常见）
        self._feat_dim_locked = False

        # TAG head
        self.proj = nn.Linear(self.feat_dim, self.feat_dim)
        self.tag_cls = nn.Linear(self.feat_dim, self.num_class)

        # 你要求：cls 权重/bias 全 0 初始化，让 logits_tag 初始≈0
        nn.init.zeros_(self.tag_cls.weight)
        nn.init.zeros_(self.tag_cls.bias)

    # -------- epoch hook (processor 每个 epoch 调一次) --------
    def set_epoch(self, epoch: int):
        self.cur_epoch = int(epoch)

    def _warmup_ramp(self) -> float:
        if self.tag_warmup_epochs <= 0:
            return 1.0
        e = float(self.cur_epoch)
        return max(0.0, min(1.0, e / float(self.tag_warmup_epochs)))

    def _infer_feat_dim_if_needed(self, feat: torch.Tensor):
        # feat: (N,C,T,V,M)
        if (not self._feat_dim_locked) and (feat is not None):
            C = int(feat.size(1))
            if C != self.feat_dim:
                # 重新建头（保持 tag_cls=0 初始化）
                self.feat_dim = C
                self.proj = nn.Linear(self.feat_dim, self.feat_dim).to(feat.device)
                self.tag_cls = nn.Linear(self.feat_dim, self.num_class).to(feat.device)
                nn.init.zeros_(self.tag_cls.weight)
                nn.init.zeros_(self.tag_cls.bias)
            self._feat_dim_locked = True

    def forward(self, x: torch.Tensor):
        """
        x: (N,C,T,V,M) from basketball_processed (C4)
        return logits: (N,num_class)
        """
        # ---------- backbone logits + feature ----------
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                # backbone.forward(x) -> (N,num_class)
                logits_base = self.backbone(x)

                # 取特征：优先 extract_feature（你的 st_gcn.py 有）
                if hasattr(self.backbone, "extract_feature"):
                    out_map, feat = self.backbone.extract_feature(x)  # out_map:(N,num_class,T,V,M), feat:(N,Cf,T,V,M)
                else:
                    feat = None
        else:
            logits_base = self.backbone(x)
            if hasattr(self.backbone, "extract_feature"):
                out_map, feat = self.backbone.extract_feature(x)
            else:
                feat = None

        # 如果拿不到 feat，就直接退化成纯 backbone（保证不掉 93）
        if feat is None:
            return logits_base

        # ---------- TAG person feature ----------
        # feat: (N,Cf,T,V,M)
        self._infer_feat_dim_if_needed(feat)

        if self.person_pool == "max":
            pf = feat.max(dim=2).values.max(dim=2).values  # (N,Cf,M)
        else:
            pf = feat.mean(dim=2).mean(dim=2)              # (N,Cf,M)
        pf = pf.permute(0, 2, 1).contiguous()              # (N,M,Cf)

        # ---------- build kNN adjacency from xyz root positions ----------
        # x[:,0:3,:,:,:] 是 xyz
        hip_l, hip_r = 11, 12
        root = 0.5 * (x[:, 0:3, :, hip_l, :] + x[:, 0:3, :, hip_r, :])  # (N,3,T,M)
        pos = root.mean(dim=2).permute(0, 2, 1).contiguous()            # (N,M,3)

        A = _build_knn_adj(pos, k=self.k, self_loop=self.self_loop)      # (N,M,M)

        # ---------- message passing ----------
        h = self.proj(pf)                                               # (N,M,Cf)
        h = h + self.lambda_fuse * torch.bmm(A, h)                       # (N,M,Cf)
        h = F.relu(h)

        # ---------- delta logits ----------
        logits_tag = self.tag_cls(h).mean(dim=1)                         # (N,num_class)

        # ---------- gate + warmup ----------
        gate = torch.sigmoid(self.tag_scale) * self.tag_scale_max        # scalar
        ramp = self._warmup_ramp()                                       # float
        logits = logits_base + (gate * ramp) * logits_tag

        return logits
