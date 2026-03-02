#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGA-INTERACT 3D skeleton clip animation (CLEAN VERSION)

- 输入: data/basketball/joints/<CLIP>_pose.npy (dict: host_*/guest_* -> (T,17,3))
- 可选动作信息: annots/tactic/<CLIP>_tactic.pkl （若存在, 标题显示动作名）
- 输出: 单个 mp4 或 gif, 画面干净, 只显示动作名 + 时长, 全屏骨骼

用法示例：
python tools/viz_clip_anim_sga17_clean.py \
    --root data/basketball \
    --clip S1_05_000754_000759 \
    --out work_dir/anim_clean \
    --fps 50 --seconds 3.2 --fmt mp4 --dpi 300
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

# ---------- 颜色 ----------
COLOR_HOST   = "#ff7f0e"   # 橙
COLOR_GUEST  = "#1f77b4"   # 蓝
COLOR_UNKNOWN= "#888888"

# ---------- 17点骨骼连接 (索引 0..16, 对应你给的图) ----------
BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),       # 头/脸
    (0, 5), (0, 6),                       # 脖子 -> 双肩
    (5, 7), (7, 9),                       # 右臂 (肩→上臂→腕)
    (6, 8), (8, 10),                      # 左臂
    (11, 12),                             # 髋
    (11, 13), (12, 14),                   # 大腿
    (13, 15), (14, 16)                    # 小腿 -> 踝
]


# ---------- 工具函数 ----------

def np_load_any(path: Path):
    """兼容 npy(pickle) / pkl"""
    try:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
            return arr.item()
        return arr
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def sort_player_keys(keys):
    """host_* 在前, 然后 guest_*; 再按数字排序"""
    def key_fn(s):
        s = s.lower()
        pri = 0 if s.startswith("host") else 1
        try:
            idx = int(s.split("_", 1)[1])
        except Exception:
            idx = 9999
        return (pri, idx, s)
    return sorted(keys, key=key_fn)


def color_of(pid: str):
    s = pid.lower()
    if s.startswith("host"):
        return COLOR_HOST
    if s.startswith("guest"):
        return COLOR_GUEST
    return COLOR_UNKNOWN


def build_xyz_from_joints_dict(jdict):
    """
    jdict: {'host_4': (T,17,3), ...}
    返回:
      xyz: (C,T,V,M)  float, 可能有 NaN
      player_keys: list[str] 长度 M
    """
    keys = sort_player_keys(list(jdict.keys()))
    arrays = [np.asarray(jdict[k]) for k in keys]
    # 对齐 T, V, C
    T = max(a.shape[0] for a in arrays)
    V = arrays[0].shape[1]
    C = arrays[0].shape[2]
    padded = []
    for a in arrays:
        a = a.astype(float)
        if a.shape[0] < T:
            pad = np.full((T - a.shape[0], V, C), np.nan, float)
            a = np.concatenate([a, pad], axis=0)
        padded.append(a)
    # (M,T,V,C)
    MTVC = np.stack(padded, axis=0)
    # (C,T,V,M)
    xyz = MTVC.transpose(3, 1, 2, 0)
    return xyz, keys


def tight_lim(arr, pad_ratio=0.05):
    """给一个 1D 数组求紧边界, 多加一点点 padding."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    span = max(1e-6, hi - lo)
    pad = pad_ratio * span
    return (lo - pad, hi + pad)


def choose_T_use(T, fps, seconds):
    """决定最终使用多少帧."""
    if seconds is None or seconds <= 0:
        return T
    T_est = int(round(seconds * fps))
    return max(1, min(T, T_est))


def load_action_label(root: Path, clip: str):
    """
    试着从 annots/tactic/<clip>_tactic.pkl 里读动作名.
    如果失败, 返回 "".
    """
    p = root / "annots" / "tactic" / f"{clip}_tactic.pkl"
    if not p.exists():
        return ""
    try:
        with open(p, "rb") as f:
            obj = pickle.load(f)
    except Exception:
        return ""

    # 多种情况尽量兼容
    if isinstance(obj, dict):
        if "Action" in obj:
            act = obj["Action"]
            if isinstance(act, dict):
                # 例如 {'Cross': [[...]], 'Isolation': [[...]]}
                keys = [k for k, v in act.items()
                        if not hasattr(v, "__len__") or len(v) > 0]
                return "/".join(keys)
            elif isinstance(act, str):
                return act
        # 其他情况: 把所有键名拼起来
        return "/".join(map(str, obj.keys()))
    return str(obj)


# ---------- 主动画函数 ----------

def animate_clip(root: Path, clip: str, outdir: Path,
                 fps: int = 50,
                 seconds: float = None,
                 fmt: str = "mp4",
                 dpi: int = 300,
                 elev: float = 25.0,
                 azim: float = -45.0):
    """
    根据 joints/<clip>_pose.npy 生成 3D 骨骼动画 (clean).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    jpath = root / "joints" / f"{clip}_pose.npy"
    assert jpath.exists(), f"找不到 joints 文件: {jpath}"

    raw = np_load_any(jpath)
    assert isinstance(raw, dict), f" joints 文件应为 dict, 但得到 {type(raw).__name__}"

    xyz, player_keys = build_xyz_from_joints_dict(raw)  # (C,T,V,M)
    C, T, V, M = xyz.shape

    T_use = choose_T_use(T, fps, seconds)
    xyz_use = xyz[:, :T_use, :, :]  # (C,T_use,V,M)

    # 预计算边界
    coords = xyz_use.reshape(C, -1)
    xlim = tight_lim(coords[0])
    ylim = tight_lim(coords[1])
    if C >= 3:
        zlim = tight_lim(coords[2])
    else:
        # 如果没有 z, 给个薄一点的盒子
        zlim = (0.0, 1.0)

    # 图像与坐标轴
    fig = plt.figure(figsize=(6, 6), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    # 全屏展示: 去掉所有白边
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    ax.view_init(elev=elev, azim=azim)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # 立体比例 (x,y,z) 也尽量均匀
    ax.set_box_aspect([
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        max(1e-6, zlim[1] - zlim[0])
    ])
    # 调整布局让骨骼占满画面
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0,0,1,1])

    # ---- 在画面内部写动作名 + 时长（不会被裁掉） ----
    action_name = load_action_label(root, clip)
    duration_text = f"{T_use}f ≈ {T_use / fps:.2f}s"
    if action_name:
        title_str = f"{action_name} | {duration_text}"
    else:
        title_str = duration_text

    # 画在左上角一点点的位置（figure 坐标系 0~1）
    fig.text(
        0.01, 0.98,               # (x, y) 越靠近 1 越靠右/上
        title_str,
        ha="left", va="top",
        fontsize=12,
        color="black",
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.7,
        ),
    )


    # 有些 mpl 版本支持 .dist, 可以稍微拉近一点
    try:
        ax.dist = 7  # 默认 10, 数值越小越近
    except Exception:
        pass
    # 为每个选手创建 scatter + bone lines
    scatters = []
    bone_lines = []  # 形状: [M][len(BONES)]

    # 初始帧
    t0 = 0
    pts0 = np.transpose(xyz_use[:, t0, :, :], (1, 2, 0))  # (V,M,C)

    for m in range(M):
        col = color_of(player_keys[m])
        p = pts0[:, m, :]
        if C >= 3:
            xs, ys, zs = p[:, 0], p[:, 1], p[:, 2]
        else:
            xs, ys, zs = p[:, 0], p[:, 1], np.zeros(V)
        sc = ax.scatter(xs, ys, zs,
                        s=18, color=col,
                        depthshade=False)
        scatters.append(sc)

        lines_m = []
        for (i, j) in BONES:
            ln, = ax.plot([], [], [], color="k", lw=2.0, alpha=0.95)
            lines_m.append(ln)
        bone_lines.append(lines_m)

    # 更新函数
    def update(frame):
        # xyz_use: (C,T_use,V,M) -> 当前帧 (V,M,C)
        pts = np.transpose(xyz_use[:, frame, :, :], (1, 2, 0))

        for m in range(M):
            p = pts[:, m, :]
            if C >= 3:
                xs, ys, zs = p[:, 0], p[:, 1], p[:, 2]
            else:
                xs, ys, zs = p[:, 0], p[:, 1], np.zeros(V)

            mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs)
            scatters[m]._offsets3d = (xs[mask], ys[mask], zs[mask])

            # 骨骼连接
            for k, (i, j) in enumerate(BONES):
                if i >= V or j >= V:
                    bone_lines[m][k].set_data_3d([], [], [])
                    continue
                if (not np.isfinite(p[i]).all()) or (not np.isfinite(p[j]).all()):
                    bone_lines[m][k].set_data_3d([], [], [])
                    continue
                seg = np.stack([p[i], p[j]], axis=0)
                if C >= 3:
                    bone_lines[m][k].set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
                else:
                    bone_lines[m][k].set_data_3d(seg[:, 0], seg[:, 1], np.zeros(2))

        return scatters + sum(bone_lines, [])

    # 创建动画
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T_use,
        interval=1000.0 / fps,
        blit=False
    )

    outdir.mkdir(parents=True, exist_ok=True)
    if fmt == "mp4":
        out_path = outdir / f"{clip}_clean.mp4"
        try:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=fps, bitrate=-1)
            ani.save(out_path.as_posix(), writer=writer, dpi=dpi)
            print(f"[OK] saved mp4: {out_path}")
            plt.close(fig)
            return
        except Exception as e:
            print(f"[warn] ffmpeg 导出失败 ({e}), 改用 GIF")
    
    # 回退 GIF
    out_path = outdir / f"{clip}_clean.gif"
    ani.save(out_path.as_posix(), writer="pillow", dpi=dpi)
    print(f"[OK] saved gif: {out_path}")
    plt.close(fig)


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="SGA-INTERACT 根目录, 如 data/basketball")
    ap.add_argument("--clip", required=True,
                    help="clip ID, 如 S1_05_000754_000759")
    ap.add_argument("--out", default="work_dir/anim_clean",
                    help="输出目录")
    ap.add_argument("--fps", type=int, default=50,
                    help="动画帧率 (默认 50)")
    ap.add_argument("--seconds", type=float, default=-1.0,
                    help="希望导出的时长 (秒). <=0 表示用全 T.")
    ap.add_argument("--fmt", choices=["mp4", "gif"], default="mp4",
                    help="输出格式, 默认 mp4, ffmpeg 不在则自动回退 gif")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=-45.0)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.out)

    secs = None if args.seconds is None or args.seconds <= 0 else args.seconds

    animate_clip(
        root=root,
        clip=args.clip,
        outdir=outdir,
        fps=args.fps,
        seconds=secs,
        fmt=args.fmt,
        dpi=args.dpi,
        elev=args.elev,
        azim=args.azim,
    )


if __name__ == "__main__":
    main()
