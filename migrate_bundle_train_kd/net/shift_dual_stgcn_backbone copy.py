# net/shift_dual_stgcn_backbone.py
import math
import importlib
from typing import Optional, Tuple

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
        frame = f.mean(dim=3)                # (N,C,T)
        w = self.scorer(frame.transpose(1, 2))  # (N,T,1)
        w = torch.softmax(w, dim=1)          # (N,T,1)
        out = (frame * w.transpose(1, 2)).sum(dim=2)  # (N,C)
        return out


class SemanticTCN(nn.Module):
    """简化语义支路：对 (N, C_sem, T, V, M) 先均值化 V、M，再做 1D-Conv 等价 MLP"""
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
        x = self.fc(x)                                   # (N, 64)
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
        self._buf: list[torch.Tensor] = []      # 临时特征缓存
        self._feat_module: Optional[nn.Module] = None   # 自动探测到的层
        self._hook: Optional[torch.utils.hooks.RemovableHandle] = None

        # 构建骨干
        Backbone = import_obj(backbone_cls)
        self.m = Backbone(**backbone_args)

        # 加载权重
        if weights:
            sd_raw = torch.load(weights, map_location='cpu')
            sd = sd_raw.get('state_dict', sd_raw)
            info = self.m.load_state_dict(sd, strict=False)
            print(f"[STGCNBackbone] loaded {weights}: missing={len(info.missing_keys)}, unexpected={len(info.unexpected_keys)}")

        # 冻结（需要的话）
        if freeze:
            for p in self.m.parameters():
                p.requires_grad = False
            self.m.eval()

    # ---- 内部：通用 hook 回调 ----
    def _cap(self, _m, _i, o):
        x = o[0] if isinstance(o, (list, tuple)) else o
        if torch.is_tensor(x) and x.dim() == 4:  # 只收集 4D 的输出
            self._buf.append(x.detach())

    # ---- 内部：自动探测最后一个 4D 特征层 ----
    def _auto_pick_feat_module(self, x5d: torch.Tensor):
        # 临时给所有子模块挂采集 hook
        handles = []
        for m in self.m.modules():
            if isinstance(m, nn.Module):
                try:
                    h = m.register_forward_hook(self._cap)
                    handles.append(h)
                except Exception:
                    pass

        self._buf.clear()
        _ = self.m(x5d)  # 触发所有 hook

        # 关掉临时 hooks
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

        if not self._buf:
            raise RuntimeError("Auto-pick failed: no 4D feature was observed. "
                               "Please ensure you feed (N,C,T,V,M) and backbone keeps M-merge until last block.")

        # 取“最后一个”4D 输出
        feat = self._buf[-1]
        # 再次正式定位：在模块树里找能产出同一个张量的模块（按 id 匹配）
        # 简化：再注册一次全局 hook，记录最后一次触发 hook 的模块对象
        last_mod = {'obj': None}
        def _cap_with_mod(m, _i, o):
            x = o[0] if isinstance(o,(list,tuple)) else o
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
            raise RuntimeError("Auto-pick couldn't identify the producing module; "
                               "fallback failed.")

        # 注册持久化 hook
        self._buf.clear()
        self._hook = self._feat_module.register_forward_hook(self._cap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, T, V, M)
        return: (N, Cf, T', V)
        """
        assert x.dim() == 5, f"expect 5D (N,C,T,V,M), got {tuple(x.shape)}"
        N, C, T, V, M = x.size()

        # 首次：自动探测并挂上持久化 hook
        if self._feat_module is None:
            self._auto_pick_feat_module(x)

        # 正常前向
        self._buf.clear()
        _ = self.m(x)

        if not self._buf:
            # 若单层 hook 没触发（某些非常规分支），回退再走一次自动探测
            self._auto_pick_feat_module(x)
            self._buf.clear()
            _ = self.m(x)
            if not self._buf:
                raise RuntimeError("forward hook still didn't capture features after re-pick. "
                                "Please check the backbone structure.")

        f = self._buf.pop()  # 期望是 (N*M, Cf, T', V) 或 (N, Cf, T', V)
        if f.dim() != 4:
            raise RuntimeError(f"expect 4D feature, got {tuple(f.shape)}")

        # 兼容两种批量：未聚合(N*M,...) 和 已聚合(N,...)
        if f.size(0) == N * M:
            # 还原 M 维 -> 聚合 M
            Cf, Tp, Vp = f.size(1), f.size(2), f.size(3)
            f = f.view(N, M, Cf, Tp, Vp)                # (N, M, C, T', V)
            if self.person_pool == 'max':
                f = f.max(dim=1).values                 # (N, C, T', V)
            elif self.person_pool == 'mean':
                f = f.mean(dim=1)                       # (N, C, T', V)
            else:
                raise ValueError(f"unknown person_pool: {self.person_pool}")
        elif f.size(0) == N:
            # 骨干内部已对人维做过聚合，直接沿用
            pass
        else:
            # 给出更友好的提示
            raise RuntimeError(
                f"N/M mismatch: feature batch={f.size(0)}, expect {N} or {N*M} (N={N}, M={M})"
            )

        return f  # (N, Cf, T', V)



# ----------------- 主模型：TAG(可选) -> ST-GCN骨干 -> TA残差门 -> 语义 -> 分类 -----------------
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

        # 0) TAG (Soft Selector / MoE over k)
        self.tag = None                      # 兼容旧逻辑：存在就走 TAG
        self.tag_experts = None              # nn.ModuleList[TAGLayer]
        self.k_selector = None               # small MLP for alpha_k
        self.k_values = None                 # list[int]
        self.tag_warmup_epochs = 0
        self.cur_epoch = 0                   # 由 processor 每个 epoch 写入

        tag_cfg = tag or {}
        if tag_cfg.get("enable", False):
            from net.tag_layer import TAGLayer
            import torch
            import torch.nn as nn

            # --- k list: 默认 2~5（你要 2~6 就在 yaml 里写 k_list: [2,3,4,5,6]）
            k_list = tag_cfg.get("k_list", None)
            if k_list is None:
                # 兼容旧字段 k: int
                k_single = int(tag_cfg.get("k", 4))
                k_list = [k_single]
            self.k_values = [int(k) for k in k_list]

            # in_channels<4 时禁用 ball
            mode = tag_cfg.get("mode", "knn")
            use_ball = bool(tag_cfg.get("use_ball", in_channels >= 4))
            if in_channels < 4:
                use_ball = False
                mode = "knn"

            # --- experts: one TAG per k
            self.tag_experts = nn.ModuleList()
            for k in self.k_values:
                self.tag_experts.append(
                    TAGLayer(
                        mode=mode,
                        k=int(k),
                        lambda_fuse=float(tag_cfg.get('lambda_fuse', 0.05)),
                        learnable_lambda=bool(tag_cfg.get('learnable_lambda', True)),
                        self_loop=bool(tag_cfg.get('self_loop', True)),
                        norm=tag_cfg.get('norm', 'row'),
                        detach_adj=bool(tag_cfg.get('detach_adj', True)),
                        use_ball=use_ball,
                        fallback=tag_cfg.get('fallback', 'knn'),
                        tau=float(tag_cfg.get('tau', 0.35)),
                        ramp_epochs=int(tag_cfg.get('ramp_epochs', 0)),
                        alpha_selfloop=float(tag_cfg.get('alpha_selfloop', 0.5))
                    )
                )

            # --- selector config
            self.tag_warmup_epochs = int(tag_cfg.get("warmup_epochs", 10))
            self.tag_scale_max = float(tag_cfg.get("tag_scale_max", 1.0))
            self.tag_scale = nn.Parameter(torch.tensor(0.0))  # learnable gate (starts ~0)

            # selector input dim：用你 backbone 的 person feature dim
            # 你工程里通常是 256（ST-GCN 的 feature）
            feat_dim = int(tag_cfg.get("selector_in_dim", 256))

            hidden = int(tag_cfg.get("selector_hidden", 128))
            self.k_selector = nn.Sequential(
                nn.Linear(feat_dim, hidden),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, len(self.k_values))
            )

            # 关键：让初始 alpha 近似均匀（避免一上来掉到 60%）
            nn.init.zeros_(self.k_selector[-1].weight)
            nn.init.zeros_(self.k_selector[-1].bias)

            # 有 soft selector 就认为启用 TAG（forward 里用 self.tag_experts 判断）
            self.tag = True


       # 1) ST-GCN 骨干（吃全部通道）
        ba = dict(backbone_args or {})
        ba['in_channels'] = int(in_channels)                       # 骨干按“总通道”建
        ba.setdefault('num_class', int(self.num_class))
        ba.setdefault('edge_importance_weighting', True)
        ba.setdefault('graph_args', graph_args or {})              # st-gcn 只要 graph_args
        ba.pop('graph', None)                                      # 避免把 graph 误传进去

        self.backbone = STGCNBackbone(
            backbone_cls=backbone_cls,
            backbone_args=ba,
            weights=backbone_weights,
            ignore_prefix=backbone_ignore,
            freeze=freeze_backbone,
            person_pool=person_pool
        )

        # 暖启动：用 5D dummy 在“骨干所在设备”上跑一次，获得探针特征以确定 Cf
        with torch.no_grad():
            T_probe = 16
            C_probe = int(in_channels)
            V_probe = int(num_point)
            M_probe = int(num_person)
            # 取骨干 device
            try:
                device = next(self.backbone.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

            # 用 M=1 的 5D dummy 探针
            dummy = torch.zeros(1, in_channels, 16, num_point, 1, device=device)

            # 给 xyz 一个极小值，避免 TAG 把人判为缺席
            if in_channels >= 3:
                dummy[:, :3].fill_(1e-3)

            # 仅探 Cf 时可跳过 TAG；若你想让 TAG 也参与探针，就保留这一行
            if self.tag is not None:
                dummy = self.tag(dummy)

            f_probe = self.backbone(dummy)   # 期望 (1, Cf, T', V)
            Cf = int(f_probe.size(1))

        Cf = int(f_probe.size(1))

        # 2) 时间注意 + 残差门
        self.ta = LightweightTemporalAttention(Cf).to(f_probe.device) if self.use_ta else None
        self.ta_gate = (nn.Sequential(
            nn.Linear(Cf, max(1, Cf // 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, Cf // 8), 1),
            nn.Sigmoid()
        ).to(f_probe.device)) if self.use_ta else None

        # 3) 语义支路
        self.semantic = SemanticTCN(self.C_sem, out_channels=64).to(f_probe.device) if self.C_sem > 0 else None

        # 4) 分类头（Cf + 语义 64）
        sem_dim = 64 if self.semantic is not None else 0
        self.classifier = nn.Linear(Cf + sem_dim, self.num_class).to(f_probe.device)
        nn.init.normal_(self.classifier.weight, 0, math.sqrt(2. / self.num_class))
        nn.init.constant_(self.classifier.bias, 0.)


        # 5) 复制教师 FC（只覆盖前 256 列；语义列清零）
        if backbone_weights is not None:
            try:
                sd_raw = torch.load(backbone_weights, map_location='cpu')
                sd = sd_raw.get('state_dict', sd_raw)
                W = b = None
                for kW, kB in [('fcn.weight','fcn.bias'), ('fc.weight','fc.bias'), ('classifier.weight','classifier.bias')]:
                    if (kW in sd) and (kB in sd):
                        W, b = sd[kW], sd[kB]
                        break
                if W is not None:
                    if W.dim()==4:
                        W = W.squeeze(-1).squeeze(-1)  # (num_class, Cf=256)
                    with torch.no_grad():
                        self.classifier.weight.zero_()
                        k = min(self.classifier.weight.size(1), W.size(1))
                        self.classifier.weight[:, :k].copy_(W[:, :k])
                        if b is not None and b.numel()==self.classifier.bias.numel():
                            self.classifier.bias.copy_(b)
                    print(f"[Init] classifier mapped teacher FC -> student [:256] and zeroed semantic ({sem_dim}).")
            except Exception as e:
                print("[Init] teacher FC copy failed:", repr(e))

        # 6) 内置教师头（冻结，用于 KD）
        kd_cfg = kd or {}
        self.kd_enable = bool(kd_cfg.get('enable', False))
        if self.kd_enable:
            self.teacher_fc = nn.Linear(Cf, self.num_class, bias=True)
            # 从 student.classifier 的左 256 列复制（语义列不复制）
            with torch.no_grad():
                self.teacher_fc.weight.copy_(self.classifier.weight[:, :Cf])
                self.teacher_fc.bias.copy_(self.classifier.bias)
            for p in self.teacher_fc.parameters():
                p.requires_grad_(False)
        else:
            self.teacher_fc = None

        # 把可学习模块放到与 f_probe 相同设备
        dev = f_probe.device
        if self.ta is not None: self.ta.to(dev)
        if self.ta_gate is not None: self.ta_gate.to(dev)
        if self.semantic is not None: self.semantic.to(dev)
        self.classifier.to(dev)
        if self.teacher_fc is not None: self.teacher_fc.to(dev)

    # ---- 工具 ----
    @staticmethod
    def _gap_tv(f: torch.Tensor) -> torch.Tensor:
        # (N,C,T,V) -> (N,C)
        return f.mean(dim=-1).mean(dim=-1)

    def set_epoch(self, epoch: int):
        # 记录给自己（需要的话）
        self._epoch = int(epoch)
        # 传给 TAG 层做 warm-up
        if getattr(self, 'tag', None) is not None:
            self.tag._epoch = int(epoch)
            
    # ---- 前向 ----
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        返回：
          - logits: (N, num_class)
          - kd_logits（若开启 KD）: (N, num_class)
        """
        # TAG
        if self.tag is not None:
            x = self.tag(x)

        # Backbone 特征
        f = self.backbone(x)            # (N,Cf,T',V)
        v_gap = self._gap_tv(f)         # (N,Cf)

        # TA 残差门
        if self.use_ta and self.ta is not None and self.ta_gate is not None:
            v_ta = self.ta(f)                           # (N,Cf)
            g = self.ta_gate(v_gap)                     # (N,1)
            v_spa = v_gap + g * v_ta                    # (N,Cf)
        else:
            v_spa = v_gap

        # 语义
        C_sem = self.C_sem
        v = v_spa
        if C_sem > 0 and self.semantic is not None:
            x_sem = x[:, -C_sem:, :, :, :]              # (N, C_sem, T, V, M)
            v_sem = self.semantic(x_sem)                # (N,64)
            v = torch.cat([v_spa, v_sem], dim=1)        # (N, Cf+64)

        logits = self.classifier(v)                     # (N,num_class)

        kd_logits = None
        if self.teacher_fc is not None:
            kd_logits = self.teacher_fc(v_gap)          # 只用 GAP-feature 走教师头

        return logits if kd_logits is None else (logits, kd_logits)
