#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGA-INTERACT 一帧多图快照生成器（无球场底图）
输出：
  1) *_skel3d.png      —— 3D 骨骼图
  2) *_topdown_points.png   —— 2D 俯视（只有点）
  3) *_ball_star.png        —— ボールグラフ（持球者星形连线）
  4) *_knn3.png, *_knn4.png, *_knn5.png, *_knn6.png —— 距離グラフ（完全连线小组）

位置定义：每名选手的“0号关节”作为平面位置（x,y）。
持球者：若有 ball_traj.pkl → 距球最近；否则回退为“与他人平均距离最小者”。

用法:
python tools/snap_graphs_sga17.py \
  --root data/basketball \
  --clip S1_05_000754_000759 \
  --out work_dir/snapshots --frame -1 --dpi 400
"""

import argparse, pickle, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- 颜色 ----------
COLOR_HOST   = "#ff7f0e"   # 橙
COLOR_GUEST  = "#1f77b4"   # 蓝
COLOR_UNKNOWN= "#888888"
COLOR_BALL   = (1.0, 0.8, 0.0)
LINE_GRAY    = "#6b7b8c"

# ---------- 你指定的骨骼连接（索引 0..16） ----------
BONES = [
    (0,1), (0,2), (1,3), (2,4),        # 头/脸
    (0,5), (0,6),                       # 脖子到双肩
    (5,7), (7,9),                       # 右臂 (肩→上臂→腕)
    (6,8), (8,10),                      # 左臂 (肩→上臂→腕)
    (11,12),                            # 髋
    (11,13), (12,14),                   # 大腿
    (13,15), (14,16)                    # 小腿到踝
]

# ---------- 工具 ----------

def np_load_any(path: Path):
    import numpy as np, pickle
    try:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype==object and arr.shape==():
            return arr.item()
        return arr
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

def sort_player_keys(keys):
    def kf(s):
        s = s.lower()
        pri = 0 if s.startswith("host") else 1
        try:
            idx = int(s.split("_",1)[1])
        except:
            idx = 9999
        return (pri, idx, s)
    return sorted(keys, key=kf)

def choose_frame(xyz, t_arg):
    # xyz: (C,T,V,M)
    C,T,V,M = xyz.shape
    if t_arg is not None and t_arg >= 0 and t_arg < T:
        return t_arg
    # 从最后一帧往前找第一个“非全 NaN”的帧
    for t in range(T-1, -1, -1):
        if np.isfinite(xyz[:,t,:,:]).any():
            return t
    return 0

def filename_to_ids(stem):
    m = re.match(r"^(S\d+_\d+)_(\d{6})_(\d{6})", stem)
    return (m.group(1), int(m.group(2)), int(m.group(3))) if m else ("",0,0)

def load_ball_xy(root: Path, clip_stem: str, T: int):
    scene, s, e = filename_to_ids(clip_stem)
    p = root/"annots"/"ball"/f"{scene}_{s:06d}_{e:06d}_ball_traj.pkl"
    if not p.exists(): return None
    with open(p,"rb") as f: obj = pickle.load(f)
    if not isinstance(obj, dict) or not obj: return None
    # 选长度最长的轨迹
    best = None
    for k,v in obj.items():
        a = np.asarray(v)
        if a.ndim==2 and a.shape[1] in (2,3):
            if best is None or len(a) > len(best):
                best = a
    if best is None: return None
    arr = best[:,:2].astype(float)
    if arr.shape[0] < T:
        pad = np.full((T-arr.shape[0],2), np.nan)
        arr = np.vstack([arr,pad])
    return arr[:T]

def centers_from_joint0(xyz_t):  # xyz_t: (V,M,C) 单帧
    # 位置用关节0的 (x,y[,z])
    # 若关节0为 NaN，则回退到该人的所有非 NaN 关节的均值
    V,M,C = xyz_t.shape
    centers = np.full((M,C), np.nan, float)
    for m in range(M):
        j0 = xyz_t[0,m,:]
        if np.isfinite(j0).all():
            centers[m] = j0
        else:
            pts = xyz_t[:,m,:]
            mask = np.isfinite(pts).all(axis=1)
            if mask.any():
                centers[m] = pts[mask].mean(axis=0)
    return centers  # (M,C)

def nearest_idx(pt_xy, centers_xy):
    d = np.linalg.norm(centers_xy - pt_xy[None,:], axis=1)
    if not np.isfinite(d).any(): return None
    j = np.nanargmin(d)
    return int(j)

def fallback_holder(centers_xy):
    # 与他人平均距离最小者
    M = centers_xy.shape[0]
    D = np.linalg.norm(centers_xy[None,:,:] - centers_xy[:,None,:], axis=2)  # (M,M)
    D[np.isnan(D)] = np.nan
    mean_d = np.nanmean(D, axis=1)
    if not np.isfinite(mean_d).any(): return 0
    return int(np.nanargmin(mean_d))

def color_of(pid):
    s = pid.lower()
    if s.startswith("host"): return COLOR_HOST
    if s.startswith("guest"): return COLOR_GUEST
    return COLOR_UNKNOWN

# ---------- 绘图 ----------

def plot_skel3d(outpath, xyz_t, player_keys, dpi=400, elev=25, azim=-45, dot=18, lw=2.0):
    V,M,C = xyz_t.shape
    fig = plt.figure(figsize=(8,8), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")
    # 坐标范围
    xs,ys,zs = xyz_t[:,:,0], xyz_t[:,:,1], (xyz_t[:,:,2] if C==3 else np.zeros((V,M)))
    m = np.isfinite(np.stack([xs,ys,zs],axis=-1)).all(axis=-1)
    xall = xs[m]; yall = ys[m]; zall = zs[m]
    def lim(a):
        if a.size==0: return (-1,1)
        lo, hi = float(np.min(a)), float(np.max(a))
        if abs(hi-lo) < 1e-6: hi = lo + 1.0
        pad = 0.1*(hi-lo)
        return (lo-pad, hi+pad)
    ax.set_xlim(lim(xall)); ax.set_ylim(lim(yall)); ax.set_zlim(lim(zall))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect([1,1,0.5])

    for m in range(M):
        col = color_of(player_keys[m])
        pts = xyz_t[:,m,:]
        ok = np.isfinite(pts).all(axis=1)
        ax.scatter(pts[ok,0], pts[ok,1], (pts[ok,2] if C==3 else np.zeros(np.sum(ok))),
                   s=dot, color=col, depthshade=False)
        for (i,j) in BONES:
            if i<V and j<V and np.isfinite(pts[[i,j]]).all():
                seg = pts[[i,j],:]
                zseg = seg[:,2] if C==3 else np.zeros(2)
                ax.plot(seg[:,0], seg[:,1], zseg, color="k", lw=lw, alpha=0.95)
    fig.savefig(outpath, transparent=True, bbox_inches="tight")
    plt.close(fig)

def _base_topdown(ax, centers_xy, player_keys, dot=140, label=True):
    M = centers_xy.shape[0]
    for m in range(M):
        if not np.isfinite(centers_xy[m]).all(): continue
        col = color_of(player_keys[m])
        ax.scatter(centers_xy[m,0], centers_xy[m,1], s=dot, edgecolors="white",
                   linewidths=1.5, color=col, zorder=3)
        if label:
            ax.text(centers_xy[m,0], centers_xy[m,1], f"{m}", color="white",
                    fontsize=10, weight="bold", ha="center", va="center", zorder=4)

def _lims2d(points):
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts)==0: return (-1,1),(-1,1)
    xmin,xmax = float(np.min(pts[:,0])), float(np.max(pts[:,0]))
    ymin,ymax = float(np.min(pts[:,1])), float(np.max(pts[:,1]))
    if xmax-xmin < 1e-6: xmax = xmin+1
    if ymax-ymin < 1e-6: ymax = ymin+1
    padx, pady = 0.1*(xmax-xmin), 0.1*(ymax-ymin)
    return (xmin-padx, xmax+padx), (ymin-pady, ymax+pady)

def plot_topdown_points(outpath, centers_xy, player_keys, dpi=400):
    fig,ax = plt.subplots(figsize=(8,8), dpi=dpi)
    _base_topdown(ax, centers_xy, player_keys, dot=140, label=True)
    xlim,ylim = _lims2d(centers_xy)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.savefig(outpath, transparent=True, bbox_inches="tight")
    plt.close(fig)

def plot_ball_star(outpath, centers_xy, player_keys, holder_idx, dpi=400):
    fig,ax = plt.subplots(figsize=(8,8), dpi=dpi)
    _base_topdown(ax, centers_xy, player_keys, dot=140, label=True)
    # 星形连线
    if holder_idx is not None and np.isfinite(centers_xy[holder_idx]).all():
        O = centers_xy[holder_idx]
        for m in range(len(player_keys)):
            if m==holder_idx or not np.isfinite(centers_xy[m]).all(): continue
            ax.plot([O[0], centers_xy[m,0]], [O[1], centers_xy[m,1]],
                    color=LINE_GRAY, lw=3, alpha=0.7, zorder=2)
        # 高亮中心
        ax.scatter(O[0],O[1], s=220, marker="*", color=COLOR_BALL,
                   edgecolors="k", linewidths=0.8, zorder=4)
    xlim,ylim = _lims2d(centers_xy)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.savefig(outpath, transparent=True, bbox_inches="tight")
    plt.close(fig)

def plot_knn_complete(outpath, centers_xy, player_keys, holder_idx, k_base, dpi=400):
    """
    k_base = 2,3,4,5 → 标题显示 k=k_base+1 (组大小)
    """
    fig,ax = plt.subplots(figsize=(8,8), dpi=dpi)
    _base_topdown(ax, centers_xy, player_keys, dot=140, label=True)
    if holder_idx is not None and np.isfinite(centers_xy[holder_idx]).all():
        O = centers_xy[holder_idx]
        # 最近 k_base 个成员（不区分队）
        diffs = centers_xy - O[None,:]
        d = np.linalg.norm(diffs, axis=1)
        d[holder_idx] = np.inf
        nn_idx = np.argsort(d)[:k_base]
        group = [holder_idx] + [int(i) for i in nn_idx if np.isfinite(centers_xy[int(i)]).all()]
        # 完全图
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                a = centers_xy[group[i]]; b = centers_xy[group[j]]
                if np.isfinite(a).all() and np.isfinite(b).all():
                    ax.plot([a[0], b[0]], [a[1], b[1]],
                            color=LINE_GRAY, lw=4, alpha=0.8, zorder=2)
        # 标题 k 显示 +1
        ax.set_title(f"k={k_base+1}", loc="left", fontsize=14, weight="bold")
    xlim,ylim = _lims2d(centers_xy)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    fig.savefig(outpath, transparent=True, bbox_inches="tight")
    plt.close(fig)

# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="SGA 数据根目录，如 data/basketball")
    ap.add_argument("--clip", required=True, help="clip id，如 S1_05_000754_000759")
    ap.add_argument("--out",  default="work_dir/snapshots")
    ap.add_argument("--frame", type=int, default=-1, help="-1=自动选最后有效帧")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=-45.0)
    args = ap.parse_args()

    root = Path(args.root)
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    stem = args.clip

    # joints
    jpath = root/"joints"/f"{stem}_pose.npy"
    raw = np_load_any(jpath)
    assert isinstance(raw, dict), f"期望 dict joints, 得到 {type(raw).__name__}"
    keys = sort_player_keys(list(raw.keys()))
    arrays = [np.asarray(raw[k]) for k in keys]  # list of (T,17,2/3)
    T = max(a.shape[0] for a in arrays)
    V = arrays[0].shape[1]; C = arrays[0].shape[2]
    # pad 对齐
    pads = []
    for a in arrays:
        if a.shape[0] < T:
            pad = np.full((T-a.shape[0], V, C), np.nan, dtype=float)
            a = np.concatenate([a, pad], axis=0)
        pads.append(a.astype(float))
    TVCM = np.stack(pads, axis=-1)          # (T,17,C,M)
    xyz = TVCM.transpose(2,0,1,3)           # (C,T,V,M)

    # 选帧
    t = choose_frame(xyz, args.frame if args.frame>=0 else None)
    C,T,V,M = xyz.shape
    xyz_t = np.transpose(xyz[:, t, :, :], (1, 2, 0))   # (V, M, C)
    centers = centers_from_joint0(xyz_t)    # (M,C)
    centers_xy = centers[:,:2].astype(float)

    # ball & holder
    ball = load_ball_xy(root, stem, T)
    if ball is not None and np.isfinite(ball[t]).all():
        holder = nearest_idx(ball[t,:2], centers_xy)
    else:
        holder = fallback_holder(centers_xy)

    # --- 输出 6 张图 ---
    # 1) 3D 骨骼
    plot_skel3d(outd/f"{stem}_t{t:03d}_skel3d.png", xyz_t, keys, dpi=args.dpi,
                elev=args.elev, azim=args.azim, dot=18, lw=2.2)

    # 2) 2D 俯视（点）
    plot_topdown_points(outd/f"{stem}_t{t:03d}_topdown_points.png",
                        centers_xy, keys, dpi=args.dpi)

    # 3) ボールグラフ（星形）
    plot_ball_star(outd/f"{stem}_t{t:03d}_ball_star.png",
                   centers_xy, keys, holder_idx=holder, dpi=args.dpi)

    # 4) 距離グラフ：k=3..6（内部基数=2..5）
    for k_base in [2,3,4,5]:
        plot_knn_complete(outd/f"{stem}_t{t:03d}_knn{k_base+1}.png",
                          centers_xy, keys, holder_idx=holder,
                          k_base=k_base, dpi=args.dpi)

    print("[OK] 输出完成：", outd.as_posix())

if __name__ == "__main__":
    main()
