# heads/dual_shift_from_feats.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ====== 你代码里的时间注意（原样复用到特征图维度） ======
class LightweightTemporalAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        hidden = max(1, in_channels // reduction_ratio)
        self.scorer = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):  # x: (N,C,T,V)
        N, C, T, V = x.shape
        frame_feat = x.mean(dim=3)               # (N,C,T)
        scores = self.scorer(frame_feat.transpose(1, 2))   # (N,T,1)
        weights = torch.softmax(scores, dim=1)             # (N,T,1)
        seq = (frame_feat * weights.transpose(1, 2)).sum(dim=2)  # (N,C)
        return seq

# ====== 轻量 T-Shift + TCN（对特征图做时域可分离卷积） ======
class TemporalShift(nn.Module):
    def __init__(self, fold_div=8): super().__init__(); self.d = fold_div
    def forward(self, x):  # (N,C,T,V)
        N, C, T, V = x.shape; f = max(1, C // self.d)
        y = x.clone()
        y[:, :f,       1:, :] = x[:, :f,       :-1, :]
        y[:, f:2*f,    :-1,:] = x[:, f:2*f,     1:, :]
        return y

class DepthwiseTCN(nn.Module):
    def __init__(self, C, k=9, r=4):
        super().__init__()
        self.ts = TemporalShift()
        self.dw = nn.Conv2d(C, C, (k,1), padding=(k//2,0), groups=C, bias=False)
        self.pw = nn.Conv2d(C, C, 1, bias=False)
        self.bn = nn.BatchNorm2d(C)
        self.se1 = nn.Conv2d(C, C//r, 1)
        self.se2 = nn.Conv2d(C//r, C, 1)
    def forward(self, x):  # (N,C,T,V)
        r = x
        x = self.ts(x)
        x = self.dw(x); x = self.pw(x); x = self.bn(x)
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.se1(w)); w = torch.sigmoid(self.se2(w))
        return F.relu(x * w) + r

class SpatialMHSA(nn.Module):
    """对 V 维做多头注意力（逐时间步），与骨架图无关，适配特征图"""
    def __init__(self, C, heads=4, attn_drop=0.0):
        super().__init__()
        self.h = heads; self.scale = (C // heads) ** -0.5
        self.qkv = nn.Conv2d(C, 3*C, 1, bias=False)
        self.proj = nn.Conv2d(C, C, 1, bias=False)
        self.drop = nn.Dropout(attn_drop)
    def forward(self, x):  # (N,C,T,V)
        N,C,T,V = x.shape
        q,k,v = torch.chunk(self.qkv(x), 3, 1)  # (N,C,T,V)
        Ch = C // self.h
        def rs(t): return t.view(N,self.h,Ch,T,V).permute(0,1,3,2,4)  # (N,h,T,Ch,V)
        q,k,v = map(rs, (q,k,v))
        attn = torch.matmul(q.transpose(-2,-1)*self.scale, k)  # (N,h,T,V,V)
        attn = F.softmax(attn, dim=-1); attn = self.drop(attn)
        out = torch.matmul(attn, v.transpose(-2,-1))          # (N,h,T,V,Ch)
        out = out.transpose(-2,-1).permute(0,1,3,2,4).contiguous().view(N,C,T,V)
        return self.proj(out)

# ====== 仅吃特征图的“双流 Shift 头” ======
class DualShiftHeadFromFeats(nn.Module):
    """
    输入：
      - x_main: (N, C1, T, V)    # 例如 ST-GCN 在 C3 数据上抽的特征图
      - x_aux:  (N, C2, T, V)    # 可选，例如 C4 特征图；不传则单流
    输出：logits (N, num_class)

    结构：
      每流：BN -> (T-Shift+DW-TCN) -> 空间MHSA
      融合：门控（单流=1门；双流=2门）-> GAP(T,V) -> FC
      时间注意：可选地对“主流”的特征图做 LightweightTemporalAttention 替代 GAP
    """
    def __init__(self, C_in, V, num_class, has_aux=False, C_aux=None,
                 heads=4, tcn_blocks=2, drop=0.2, use_temporal_attention=True):
        super().__init__()
        self.has_aux = has_aux
        self.use_temporal_attention = use_temporal_attention

        self.norm1 = nn.BatchNorm2d(C_in)
        self.tcn1  = nn.Sequential(*[DepthwiseTCN(C_in) for _ in range(tcn_blocks)])
        self.attn1 = SpatialMHSA(C_in, heads=heads)
        if use_temporal_attention:
            self.tattn1 = LightweightTemporalAttention(C_in)

        if has_aux:
            assert C_aux is not None
            self.norm2 = nn.BatchNorm2d(C_aux)
            self.tcn2  = nn.Sequential(*[DepthwiseTCN(C_aux) for _ in range(tcn_blocks)])
            self.attn2 = SpatialMHSA(C_aux, heads=heads)
            if use_temporal_attention:
                self.tattn2 = LightweightTemporalAttention(C_aux)

        C_gate = C_in + (C_aux if has_aux else 0)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C_gate, max(1, C_gate//4), 1), nn.ReLU(True),
            nn.Conv2d(max(1, C_gate//4), 2 if has_aux else 1, 1)
        )
        self.drop = nn.Dropout(drop)
        self.fc   = nn.Linear(C_in if not has_aux else (C_in + C_aux), num_class)

    def _branch(self, x, norm, tcn, attn, tattn=None):  # x: (N,C,T,V)
        x = norm(x)
        xt = tcn(x)
        xs = attn(x)
        f  = xt + xs
        if tattn is not None:
            # 你论文里的“时间注意”分支（对特征图做帧加权得到向量）
            v = tattn(f)   # (N,C)
        else:
            v = f.mean(-1).mean(-1)  # GAP
        return f, v  # f: (N,C,T,V), v: (N,C)

    def forward(self, x1, x2=None):
        f1, v1 = self._branch(x1, self.norm1, self.tcn1, self.attn1, self.tattn1 if self.use_temporal_attention else None)
        if not self.has_aux or x2 is None:
            g = torch.sigmoid(self.gate(f1).squeeze(-1).squeeze(-1))  # (N,1)
            f = f1 * g[:, None, None, None]
            h = self.drop(v1)  # 向量路径更稳
            return self.fc(h)

        f2, v2 = self._branch(x2, self.norm2, self.tcn2, self.attn2, self.tattn2 if self.use_temporal_attention else None)
        z = torch.cat([f1, f2], dim=1)
        g = torch.softmax(self.gate(z).squeeze(-1).squeeze(-1), dim=1)  # (N,2)
        v = torch.cat([v1, v2], dim=1)  # (N,C1+C2)
        v = self.drop(v)
        return self.fc(v)
