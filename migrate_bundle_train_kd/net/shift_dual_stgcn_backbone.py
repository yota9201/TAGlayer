# net/shift_dual_stgcn_backbone.py
import math
import importlib
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def import_obj(qualname: str):
    """
    e.g. 'net.st_gcn.Model' -> 返回 net.st_gcn 里的 Model
         'net.st_gcn:Model' 也支持
    """
    qualname = qualname.replace(':', '.')
    mod_name, attr = qualname.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


# ----------------- 小组件 -----------------
class LightweightTemporalAttention(nn.Module):
    """输入: (N, C, T, V) -> 输出: (N, C) 加权时间汇聚"""
    def __init__(self, in_channels: int, hidden_ratio: int = 8):
        super().__init__()
        h = max(1, in_channels // hidden_ratio)
        self.scorer = nn.Sequential(
            nn.Linear(in_channels, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, 1)
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: (N,C,T,V)
        n, c, t, v = f.shape
        frame = f.mean(dim=3)                  # (N,C,T)
        w = self.scorer(frame.transpose(1, 2)) # (N,T,1)
        w = torch.softmax(w, dim=1)            # (N,T,1)
        out = (frame * w.transpose(1, 2)).sum(dim=2)  # (N,C)
        return out


class SemanticTCN(nn.Module):
    """简化语义支路：对 (N, C_sem, T, V, M) 先均值化 V、M、T，再做 BN+FC"""
    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.fc = nn.Linear(in_channels, out_channels)
        nn.init.constant_(self.bn.weight, 1.)
        nn.init.constant_(self.bn.bias, 0.)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / out_channels))
        nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x_sem: torch.Tensor) -> torch.Tensor:
        # x_sem: (N, C_sem, T, V, M)
        if x_sem is None:
            raise RuntimeError("x_sem is None")
        x = x_sem.mean(dim=-1).mean(dim=-1).mean(dim=2)  # (N, C_sem)
        x = self.bn(x)
        x = self.fc(x)  # (N, 64)
        return x


# ----------------- ST-GCN 骨干封装 -----------------
class STGCNBackbone(nn.Module):
    """
    包一层 ST-GCN:
    - 首次 forward 自动探测并记住“最后一个 4D 特征层”（形状 ~ (N*M, C, T', V)）
    - 之后仅对该层挂 hook，抓取特征，然后按 person_pool 聚合到 (N, C, T', V)
    """
    def __init__(self, backbone_cls, backbone_args, weights: Optional[str]=None,
                 ignore_prefix=None, freeze: bool=True, person_pool: str='max'):
        super().__init__()
        self.person_pool = person_pool
        self._buf: List[torch.Tensor] = []
        self._feat_module: Optional[nn.Module] = None
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = None

        Backbone = import_obj(backbone_cls)
        self.m = Backbone(**backbone_args)

        if weights:
            sd_raw = torch.load(weights, map_location='cpu')
            sd = sd_raw.get('state_dict', sd_raw)
            info = self.m.load_state_dict(sd, strict=False)
            print(f"[STGCNBackbone] loaded {weights}: missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")

        if freeze:
            for p in self.m.parameters():
                p.requires_grad = False
            self.m.eval()

    def _cap(self, _m, _i, o):
        x = o[0] if isinstance(o, (list, tuple)) else o
        if torch.is_tensor(x) and x.dim() == 4:
            self._buf.append(x.detach())

    def _auto_pick_feat_module(self, x5d: torch.Tensor):
        handles = []
        for m in self.m.modules():
            try:
                handles.append(m.register_forward_hook(self._cap))
            except Exception:
                pass

        self._buf.clear()
        _ = self.m(x5d)

        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

        if not self._buf:
            raise RuntimeError("Auto-pick failed: no 4D feature was observed.")

        last_mod = {'obj': None}

        def _cap_with_mod(m, _i, o):
            x = o[0] if isinstance(o, (list, tuple)) else o
            if torch.is_tensor(x) and x.dim() == 4:
                last_mod['obj'] = m

        handles = []
        for m in self.m.modules():
            try:
                handles.append(m.register_forward_hook(_cap_with_mod))
            except Exception:
                pass

        _ = self.m(x5d)

        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

        self._feat_module = last_mod['obj']
        if self._feat_module is None:
            raise RuntimeError("Auto-pick couldn't identify producing module.")

        self._buf.clear()
        self._hook = self._feat_module.register_forward_hook(self._cap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, T, V, M) -> (N, Cf, T', V)
        """
        assert x.dim() == 5, f"expect 5D (N,C,T,V,M), got {tuple(x.shape)}"
        N, C, T, V, M = x.size()
        # --- Robust channel alignment for loading 3/4ch ST-GCN weights with 5ch inputs ---
        # Infer expected input channels from underlying ST-GCN data_bn: num_features = in_channels * V.
        expected_c = None
        try:
            if hasattr(self, 'm') and hasattr(self.m, 'data_bn') and hasattr(self.m, 'A'):
                Vg = int(self.m.A.size(1))  # number of joints
                nf = int(self.m.data_bn.num_features)
                if Vg > 0 and nf % Vg == 0:
                    expected_c = nf // Vg
        except Exception:
            expected_c = None
        if expected_c is not None and C != expected_c:
            if C < expected_c:
                raise RuntimeError(f"input has C={C} < expected backbone channels {expected_c}")
            x = x[:, :expected_c, :, :, :]
            N, C, T, V, M = x.size()

        if self._feat_module is None:
            self._auto_pick_feat_module(x)

        self._buf.clear()
        _ = self.m(x)

        if not self._buf:
            self._auto_pick_feat_module(x)
            self._buf.clear()
            _ = self.m(x)
            if not self._buf:
                raise RuntimeError("forward hook still didn't capture features.")

        f = self._buf.pop()
        if f.dim() != 4:
            raise RuntimeError(f"expect 4D feature, got {tuple(f.shape)}")

        if f.size(0) == N * M:
            Cf, Tp, Vp = f.size(1), f.size(2), f.size(3)
            f = f.view(N, M, Cf, Tp, Vp)  # (N, M, C, T', V)
            if self.person_pool == 'max':
                f = f.max(dim=1).values
            elif self.person_pool == 'mean':
                f = f.mean(dim=1)
            else:
                raise ValueError(f"unknown person_pool: {self.person_pool}")
        elif f.size(0) == N:
            pass
        else:
            raise RuntimeError(f"N/M mismatch: feature batch={f.size(0)}, expect {N} or {N*M}")

        return f  # (N, Cf, T', V)


# ----------------- 主模型：(SoftSelector TAG可选) -> ST-GCN骨干 -> TA门 -> 语义 -> 分类 -> KD -----------------
class Model(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 num_class: int = 20,
                 num_point: int = 17,
                 num_person: int = 6,
                 graph: str = 'net.graph.Graph',
                 graph_args: dict | None = None,
                 # TAG
                 tag: dict | None = None,
                 # Backbone
                 backbone_cls: str = 'net.st_gcn.Model',
                 backbone_args: dict | None = None,
                 backbone_weights: str | None = None,
                 backbone_ignore: list[str] | None = None,
                 freeze_backbone: bool = True,
                 person_pool: str = 'mean',
                 # semantic / TA
                 in_channels_semantic: int = 1,
                 use_temporal_attention: bool = True,
                 # KD
                 kd: dict | None = None):
        super().__init__()
        self.num_class = int(num_class)
        self.C_sem = int(in_channels_semantic)
        self.use_ta = bool(use_temporal_attention)

        # ========= 0) Soft-Selector TAG (MoE over k) =========
        self.tag_enable = False
        self.tag_mode = "knn"
        self.tag_use_ball = bool(in_channels >= 4)
        self.tag_experts: Optional[nn.ModuleList] = None
        self.k_values: Optional[List[int]] = None
        self.k_selector: Optional[nn.Module] = None

        # gate + warmup
        self.tag_gate = nn.Parameter(torch.tensor(0.0))   # sigmoid gate, init ~0
        self.tag_gate_max = 1.0
        self.tag_warmup_epochs = 0
        self._epoch = 0

        tag_cfg = tag or {}
        if tag_cfg.get("enable", False):
            from net.tag_layer import TAGLayer

            self.tag_enable = True
            self.tag_mode = str(tag_cfg.get("mode", "knn"))
            self.tag_use_ball = bool(tag_cfg.get("use_ball", in_channels >= 4))
            if in_channels < 4:
                self.tag_use_ball = False
                self.tag_mode = "knn"

            k_list = tag_cfg.get("k_list", None)
            if k_list is None:
                k_single = int(tag_cfg.get("k", 4))
                k_list = [k_single]
            self.k_values = [int(k) for k in k_list]

            self.tag_warmup_epochs = int(tag_cfg.get("warmup_epochs", tag_cfg.get("ramp_epochs", 12)))
            self.tag_gate_max = float(tag_cfg.get("tag_gate_max", tag_cfg.get("tag_scale_max", 1.0)))

            # Experts
            self.tag_experts = nn.ModuleList()
            for k in self.k_values:
                self.tag_experts.append(
                    TAGLayer(
                        mode=self.tag_mode,
                        k=int(k),
                        lambda_fuse=float(tag_cfg.get('lambda_fuse', 0.10)),
                        learnable_lambda=bool(tag_cfg.get('learnable_lambda', True)),
                        self_loop=bool(tag_cfg.get('self_loop', True)),
                        norm=tag_cfg.get('norm', 'sym'),
                        detach_adj=bool(tag_cfg.get('detach_adj', True)),
                        use_ball=self.tag_use_ball,
                        fallback=tag_cfg.get('fallback', 'knn'),
                        tau=float(tag_cfg.get('tau', 0.35)),
                        ramp_epochs=int(tag_cfg.get('ramp_epochs', 0)),
                        alpha_selfloop=float(tag_cfg.get('alpha_selfloop', 0.5)),
                        soft_selector=None,  # 这里我们在 Model 内部做 selector
                    )
                )

            # Selector (no labels needed)
            selector_type = str(tag_cfg.get("selector", "per_sample")).lower()
            if selector_type not in ["global", "per_sample"]:
                selector_type = "per_sample"
            self.selector_type = selector_type

            if self.selector_type == "global":
                self.alpha_logits = nn.Parameter(torch.zeros(len(self.k_values)))
            else:
                hidden = int(tag_cfg.get("selector_hidden", 64))
                self.k_selector = nn.Sequential(
                    nn.Linear(8, hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden, len(self.k_values))
                )
                # init uniform
                nn.init.zeros_(self.k_selector[-1].weight)
                nn.init.zeros_(self.k_selector[-1].bias)

            self.selector_temperature = float(tag_cfg.get("selector_temperature", 1.0))

        # ========= 1) ST-GCN backbone =========
        ba = dict(backbone_args or {})
        ba.setdefault('in_channels', int(in_channels))
        ba.setdefault('num_class', int(self.num_class))
        ba.setdefault('edge_importance_weighting', True)
        ba.setdefault('graph_args', graph_args or {})
        ba.pop('graph', None)

        self.backbone = STGCNBackbone(
            backbone_cls=backbone_cls,
            backbone_args=ba,
            weights=backbone_weights,
            ignore_prefix=backbone_ignore,
            freeze=freeze_backbone,
            person_pool=person_pool
        )

        # ========= probe Cf =========
        with torch.no_grad():
            try:
                device = next(self.backbone.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

            dummy = torch.zeros(1, in_channels, 16, num_point, 1, device=device)
            if in_channels >= 3:
                dummy[:, :3].fill_(1e-3)

            # 探 Cf 时，不让 TAG 破坏探针稳定性：直接走原 dummy
            f_probe = self.backbone(dummy)  # (1, Cf, T', V)
            Cf = int(f_probe.size(1))

        # ========= 2) TA =========
        self.ta = LightweightTemporalAttention(Cf).to(f_probe.device) if self.use_ta else None
        self.ta_gate = (nn.Sequential(
            nn.Linear(Cf, max(1, Cf // 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, Cf // 8), 1),
            nn.Sigmoid()
        ).to(f_probe.device)) if self.use_ta else None

        # ========= 3) semantic =========
        self.semantic = SemanticTCN(self.C_sem, out_channels=64).to(f_probe.device) if self.C_sem > 0 else None

        # ========= 4) classifier =========
        sem_dim = 64 if self.semantic is not None else 0
        self.classifier = nn.Linear(Cf + sem_dim, self.num_class).to(f_probe.device)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2. / self.num_class))
        nn.init.constant_(self.classifier.bias, 0.)

        # ========= 5) init from teacher FC (optional, keep your old behavior) =========
        if backbone_weights is not None:
            try:
                sd_raw = torch.load(backbone_weights, map_location='cpu')
                sd = sd_raw.get('state_dict', sd_raw)
                W = b = None
                for kW, kB in [('fcn.weight', 'fcn.bias'), ('fc.weight', 'fc.bias'), ('classifier.weight', 'classifier.bias')]:
                    if (kW in sd) and (kB in sd):
                        W, b = sd[kW], sd[kB]
                        break
                if W is not None:
                    if W.dim() == 4:
                        W = W.squeeze(-1).squeeze(-1)
                    with torch.no_grad():
                        self.classifier.weight.zero_()
                        k = min(self.classifier.weight.size(1), W.size(1))
                        self.classifier.weight[:, :k].copy_(W[:, :k])
                        if b is not None and b.numel() == self.classifier.bias.numel():
                            self.classifier.bias.copy_(b)
                    print(f"[Init] classifier mapped teacher FC -> student [:256] and zeroed semantic ({sem_dim}).")
            except Exception as e:
                print("[Init] teacher FC copy failed:", repr(e))

        # ========= 6) internal teacher head for KD =========
        kd_cfg = kd or {}
        self.kd_enable = bool(kd_cfg.get('enable', False))
        if self.kd_enable:
            self.teacher_fc = nn.Linear(Cf, self.num_class, bias=True)
            with torch.no_grad():
                self.teacher_fc.weight.copy_(self.classifier.weight[:, :Cf])
                self.teacher_fc.bias.copy_(self.classifier.bias)
            for p in self.teacher_fc.parameters():
                p.requires_grad_(False)
        else:
            self.teacher_fc = None

        dev = f_probe.device
        if self.ta is not None: self.ta.to(dev)
        if self.ta_gate is not None: self.ta_gate.to(dev)
        if self.semantic is not None: self.semantic.to(dev)
        self.classifier.to(dev)
        if self.teacher_fc is not None: self.teacher_fc.to(dev)

    # ---- utils ----
    @staticmethod
    def _gap_tv(f: torch.Tensor) -> torch.Tensor:
        return f.mean(dim=-1).mean(dim=-1)  # (N,C,T,V)->(N,C)

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        # 让每个 TAG expert 都拿到 epoch（用于它内部 ramp，如有）
        if self.tag_experts is not None:
            for e in self.tag_experts:
                if hasattr(e, "set_epoch"):
                    e.set_epoch(self._epoch)
                else:
                    e._epoch = self._epoch

    def _tag_ramp(self) -> float:
        if self.tag_warmup_epochs <= 0:
            return 1.0
        return max(0.0, min(1.0, float(self._epoch) / float(self.tag_warmup_epochs)))

    def _selector_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,T,V,M) -> feat: (N,8)
        不依赖 backbone，直接用几何/球权统计，保证能在 TAG 前算 alpha
        """
        N, C, T, V, M = x.shape
        # root pos (hip 11/12)
        hip_l, hip_r = 11, 12
        root = 0.5 * (x[:, 0:3, :, hip_l, :] + x[:, 0:3, :, hip_r, :])  # (N,3,T,M)
        pos = root.mean(dim=2).permute(0, 2, 1).contiguous()            # (N,M,3)
        dmean = torch.cdist(pos, pos).mean(dim=(1, 2))                  # (N,)

        if (C >= 4) and self.tag_use_ball:
            s = x[:, 3, :, :, :].mean(dim=1).mean(dim=1)  # (N,M)
            s_mean = s.mean(dim=1)
            s_std = s.std(dim=1)
            s_max = s.max(dim=1).values
            s_min = s.min(dim=1).values
        else:
            s_mean = torch.zeros(N, device=x.device, dtype=x.dtype)
            s_std  = torch.zeros(N, device=x.device, dtype=x.dtype)
            s_max  = torch.zeros(N, device=x.device, dtype=x.dtype)
            s_min  = torch.zeros(N, device=x.device, dtype=x.dtype)

        feat = torch.stack([
            dmean,
            pos[..., 0].std(dim=1),
            pos[..., 1].std(dim=1),
            pos[..., 2].std(dim=1),
            s_mean, s_std, s_max, s_min,
        ], dim=1)
        return feat  # (N,8)

    def _alpha(self, x: torch.Tensor) -> torch.Tensor:
        """
        return alpha: (N,K)
        """
        K = len(self.k_values) if self.k_values is not None else 1
        temp = max(1e-6, float(getattr(self, "selector_temperature", 1.0)))

        if (not self.tag_enable) or (self.k_values is None) or (K == 1):
            return torch.ones(x.size(0), K, device=x.device, dtype=x.dtype) / K

        if getattr(self, "selector_type", "per_sample") == "global":
            a = F.softmax(self.alpha_logits / temp, dim=0).view(1, K).expand(x.size(0), -1)
            return a.to(device=x.device, dtype=x.dtype)

        feat = self._selector_features(x)           # (N,8)
        logits = self.k_selector(feat)              # (N,K)
        a = F.softmax(logits / temp, dim=1)
        return a

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        返回：
          - logits: (N, num_class)
          - kd_logits（若开启 KD）: (N, num_class)
        """
        # ========= TAG (soft selector) =========
        if self.tag_enable and (self.tag_experts is not None) and (self.k_values is not None) and (len(self.k_values) > 0):
            # compute alpha_k
            alpha = self._alpha(x)  # (N,K)

            # run experts
            xs = []
            for expert in self.tag_experts:
                xs.append(expert(x))  # each: (N,C,T,V,M)
            x_stack = torch.stack(xs, dim=1)  # (N,K,C,T,V,M)

            # mix
            alpha_view = alpha.view(alpha.size(0), alpha.size(1), 1, 1, 1, 1)
            x_mix = (alpha_view * x_stack).sum(dim=1)  # (N,C,T,V,M)

            # gate + warmup ramp (start ~0)
            gate = torch.sigmoid(self.tag_gate) * float(self.tag_gate_max)
            ramp = float(self._tag_ramp())
            x = x + (gate * ramp) * (x_mix - x)

        # ========= Backbone feature =========
        f = self.backbone(x)         # (N,Cf,T',V)
        v_gap = self._gap_tv(f)      # (N,Cf)

        # ========= TA residual gate =========
        if self.use_ta and self.ta is not None and self.ta_gate is not None:
            v_ta = self.ta(f)            # (N,Cf)
            g = self.ta_gate(v_gap)      # (N,1)
            v_spa = v_gap + g * v_ta
        else:
            v_spa = v_gap

        # ========= semantic =========
        v = v_spa
        if self.C_sem > 0 and self.semantic is not None:
            x_sem = x[:, -self.C_sem:, :, :, :]  # (N,C_sem,T,V,M)
            v_sem = self.semantic(x_sem)         # (N,64)
            v = torch.cat([v_spa, v_sem], dim=1)

        # ========= classifier =========
        logits = self.classifier(v)  # (N,num_class)

        kd_logits = None
        if self.teacher_fc is not None:
            kd_logits = self.teacher_fc(v_gap)

        return logits if kd_logits is None else (logits, kd_logits)
