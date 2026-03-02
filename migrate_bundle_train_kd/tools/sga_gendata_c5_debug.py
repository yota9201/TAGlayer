# /workspace/st-gcn/tools/sga_gendata_c5_debug.py
# -*- coding: utf-8 -*-
"""
SGA-INTERACT → ST-GCN 数据生成（C=3/4/5） + 强化诊断版

你现在遇到的 “生成样本 0 条” 基本都来自：
- joints_path / tactic_path 找不到（最常见：data_root 指错，或子目录名不同）
- joints 读出来为空
- tactic 里 Action 为空 / intervals 为空

本脚本会：
1) 统计各类 skip 原因
2) 打印前 N 个缺失文件样例（便于你一眼发现路径不对）
3) 支持通过命令行覆盖 joints/tactic/ball 子目录（适配你本地数据结构）

输出通道：
- C=3: xyz
- C=4: xyz + ball mask
- C=5: xyz + ball mask + offense_team mask（由 ball_sum top3 推断）
"""

import os
import sys
import pickle
import argparse
import numpy as np

try:
    from tqdm import tqdm
    use_tqdm = True
except Exception:
    use_tqdm = False

FRAME_RATE = 50
MAX_FRAME  = 400
MAX_BODY   = 6
NUM_JOINT  = 17


def _parse_split_file(split_file_path):
    with open(split_file_path, 'rb') as f:
        obj = pickle.load(f)

    clips = []
    if isinstance(obj, dict):
        for _cls, items in obj.items():
            for it in items:
                sid = it[0] if isinstance(it, (list, tuple)) else it
                sid = str(sid)
                if not sid.endswith('_tactic.pkl'):
                    sid = f"{sid}_tactic.pkl"
                clips.append(sid)
    elif isinstance(obj, (list, tuple)):
        for sid in obj:
            sid = str(sid)
            if not sid.endswith('_tactic.pkl'):
                sid = f"{sid}_tactic.pkl"
            clips.append(sid)
    else:
        raise ValueError(f"不支持的 split 结构: {type(obj)} @ {split_file_path}")
    return clips


def _load_joints_any(path):
    obj = np.load(path, allow_pickle=True)

    # dict-like
    if hasattr(obj, "item"):
        try:
            d = obj.item()
            if isinstance(d, dict):
                out = {}
                for k, v in d.items():
                    arr = np.asarray(v)
                    if arr.ndim != 3 or arr.shape[2] < 3:
                        continue
                    out[str(k)] = arr[:, :NUM_JOINT, :3]
                if len(out) > 0:
                    return out
        except Exception:
            pass

    # ndarray
    arr = np.asarray(obj)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            return {'guest_1': arr[:, :NUM_JOINT, :3]}
    elif arr.ndim == 4:
        T, V, C, M = arr.shape
        out = {}
        for m in range(min(M, MAX_BODY)):
            out[f'guest_{m+1}'] = arr[:, :NUM_JOINT, :3, m]
        return out
    return {}


def _normalize_coords_(sample_cvmm):
    coords = sample_cvmm[:3]
    max_val = np.abs(coords).max()
    eps = 1e-6
    if max_val > eps:
        sample_cvmm[:3] = coords / (max_val + eps)
    return sample_cvmm


def _infer_offense_from_ball(sample_c_t_v_m, topk=3):
    M = sample_c_t_v_m.shape[-1]
    ball_sum = sample_c_t_v_m[3, :, 0, :].sum(axis=0)  # (M,)
    idx_sorted = np.argsort(-ball_sum)
    off_idx = idx_sorted[:min(topk, M)]
    if float(ball_sum.max()) <= 0.0:
        off_idx = np.arange(min(topk, M), dtype=np.int64)
    off_mask = np.zeros((M,), dtype=np.float32)
    off_mask[off_idx] = 1.0
    return off_mask


def gendata_for_stgcn(
    data_root,
    out_path,
    split_file_path,
    part,
    channels=5,
    action_map_file=None,
    joints_subdir='joints',
    tactic_subdir=os.path.join('annots', 'tactic'),
    ball_subdir=os.path.join('annots', 'ball'),
    debug_show=20
):
    assert channels in (3, 4, 5), f"channels 必须为 3/4/5，得到 {channels}"

    joints_dir = os.path.join(data_root, joints_subdir)
    tactic_dir = os.path.join(data_root, tactic_subdir)
    ball_dir   = os.path.join(data_root, ball_subdir)

    print(f"[paths] data_root={os.path.abspath(data_root)}")
    print(f"[paths] joints_dir={os.path.abspath(joints_dir)}")
    print(f"[paths] tactic_dir={os.path.abspath(tactic_dir)}")
    print(f"[paths] ball_dir  ={os.path.abspath(ball_dir)}")

    # preload ball
    all_ball_data = {}
    if channels >= 4:
        print("正在预加载所有球权标注文件...")
        if os.path.exists(ball_dir):
            for fn in os.listdir(ball_dir):
                if fn.endswith('_ball_traj.pkl'):
                    base = fn.replace('_ball_traj.pkl', '')
                    with open(os.path.join(ball_dir, fn), 'rb') as f:
                        try:
                            all_ball_data[base] = pickle.load(f)
                        except Exception:
                            all_ball_data[base] = {}
        print(f"成功加载了 {len(all_ball_data)} 个样本的球权信息。")
    else:
        print("channels=3，跳过球权标注加载。")

    player_id_to_idx = {f'guest_{i+1}': i for i in range(MAX_BODY)}

    if not os.path.exists(split_file_path):
        raise FileNotFoundError(f"Split file not found: {split_file_path}")
    list_of_clips = _parse_split_file(split_file_path)
    print(f"正在为 '{part}' 部分处理 {len(list_of_clips)} 个片段...")

    if action_map_file is None:
        action_map_file = os.path.join(out_path, 'action_map.pkl')

    action_to_id = {}
    if os.path.exists(action_map_file):
        with open(action_map_file, 'rb') as f:
            action_to_id = pickle.load(f)

    current_action_id = len(action_to_id)
    if part == 'val' and current_action_id == 0:
        raise FileNotFoundError(f"未找到 action_map.pkl：{action_map_file}；请先 train 再 val")

    sample_names = []
    sample_labels = []
    all_samples = []

    # ---- diagnostics ----
    stats = dict(
        missing_pair=0,
        missing_joints=0,
        missing_tactic=0,
        bad_joints=0,
        bad_tactic=0,
        empty_action=0,
        empty_intervals=0,
        kept_samples=0,
    )
    missing_examples = []
    bad_examples = []

    iterator = tqdm(list_of_clips, desc=f"[{part}]") if use_tqdm else list_of_clips

    for tactic_fn in iterator:
        if not (isinstance(tactic_fn, str) and tactic_fn.endswith('_tactic.pkl')):
            continue

        base = tactic_fn.replace('_tactic.pkl', '')
        joints_path = os.path.join(joints_dir, f"{base}_pose.npy")
        tactic_path = os.path.join(tactic_dir, tactic_fn)

        has_j = os.path.exists(joints_path)
        has_t = os.path.exists(tactic_path)
        if (not has_j) or (not has_t):
            stats["missing_pair"] += 1
            if not has_j: stats["missing_joints"] += 1
            if not has_t: stats["missing_tactic"] += 1
            if len(missing_examples) < debug_show:
                missing_examples.append((base, joints_path, has_j, tactic_path, has_t))
            continue

        joints_dict = _load_joints_any(joints_path)
        if len(joints_dict) == 0:
            stats["bad_joints"] += 1
            if len(bad_examples) < debug_show:
                bad_examples.append(("bad_joints", base, joints_path))
            continue

        try:
            with open(tactic_path, 'rb') as f:
                tac = pickle.load(f)
        except Exception:
            stats["bad_tactic"] += 1
            if len(bad_examples) < debug_show:
                bad_examples.append(("bad_tactic", base, tactic_path))
            continue

        action_dict = tac.get('Action', None)
        if not isinstance(action_dict, dict) or len(action_dict) == 0:
            stats["empty_action"] += 1
            continue

        for action_name, intervals in action_dict.items():
            if part == 'train' and action_name not in action_to_id:
                action_to_id[action_name] = current_action_id
                current_action_id += 1
            if action_name not in action_to_id:
                continue
            label = action_to_id[action_name]
            if not intervals:
                stats["empty_intervals"] += 1
                continue

            for inst_idx, interval in enumerate(intervals):
                if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                    continue
                start_time, end_time = interval
                start_frame = int(start_time * FRAME_RATE)
                end_frame   = int(end_time   * FRAME_RATE)
                if end_frame - start_frame <= 0:
                    continue

                sample = np.zeros((channels, MAX_FRAME, NUM_JOINT, MAX_BODY), dtype=np.float32)

                # xyz
                keys = sorted(list(joints_dict.keys()))
                for m in range(MAX_BODY):
                    if m < len(keys):
                        pid = keys[m]
                        seq = np.asarray(joints_dict[pid])
                        if seq.ndim != 3 or seq.shape[2] < 3:
                            continue
                        seg = seq[start_frame:end_frame]
                        if seg.shape[0] <= 0:
                            continue
                        t_len = min(seg.shape[0], MAX_FRAME)
                        v_len = min(seg.shape[1], NUM_JOINT)
                        sample[:3, :t_len, :v_len, m] = seg[:t_len, :v_len, :3].transpose(2, 0, 1)

                _normalize_coords_(sample)

                # ball
                if channels >= 4:
                    ball_raw = all_ball_data.get(base, {})
                    if isinstance(ball_raw, dict):
                        for player_id, traj in ball_raw.items():
                            if player_id in player_id_to_idx and hasattr(traj, '__len__'):
                                m = player_id_to_idx[player_id]
                                for frame_idx, _ in enumerate(traj):
                                    if start_frame <= frame_idx < end_frame:
                                        t_in = frame_idx - start_frame
                                        if 0 <= t_in < MAX_FRAME:
                                            sample[3, t_in, :, m] = 1.0

                # offense team
                if channels == 5:
                    off_mask = _infer_offense_from_ball(sample, topk=3)
                    for m in range(MAX_BODY):
                        sample[4, :, :, m] = off_mask[m]

                all_samples.append(sample)
                sample_labels.append(label)
                sample_names.append(f"{base}_action_{action_name}_{inst_idx}")
                stats["kept_samples"] += 1

    # ---- save ----
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, f'{part}_label.pkl'), 'wb') as f:
        pickle.dump((sample_names, list(sample_labels)), f)

    if len(all_samples) == 0:
        np.save(os.path.join(out_path, f'{part}_data.npy'),
                np.zeros((0, channels, MAX_FRAME, NUM_JOINT, MAX_BODY), dtype=np.float32))
    else:
        np.save(os.path.join(out_path, f'{part}_data.npy'),
                np.stack(all_samples, axis=0))

    if part == 'train':
        with open(action_map_file, 'wb') as f:
            pickle.dump(action_to_id, f)
        print(f"[INFO] 已保存动作映射到: {action_map_file}")

    print(f"[INFO] {part}: 生成样本 {len(all_samples)} 条；输出目录: {out_path}")

    # ---- report diagnostics ----
    print("\n========== DIAGNOSTICS ==========")
    for k, v in stats.items():
        print(f"{k:>16s}: {v}")
    if missing_examples:
        print("\n[missing examples] (base, joints_path, exists?, tactic_path, exists?)")
        for ex in missing_examples:
            print(" -", ex[0])
            print("    joints:", ex[1], "exists" if ex[2] else "MISSING")
            print("    tactic:", ex[3], "exists" if ex[4] else "MISSING")
    if bad_examples:
        print("\n[bad examples]")
        for ex in bad_examples:
            print(" -", ex)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SGA-INTERACT → ST-GCN 数据转换（诊断版）')
    parser.add_argument('--data_path', required=True, help='SGA-INTERACT 数据集根目录（raw）')
    parser.add_argument('--out_folder', required=True, help='输出根目录')
    parser.add_argument('--benchmark', default='GAR', choices=['GAR', 'GAL'])
    parser.add_argument('--channels', type=int, default=5, choices=[3, 4, 5])

    # allow overriding subdirs
    parser.add_argument('--joints_subdir', default='joints')
    parser.add_argument('--tactic_subdir', default=os.path.join('annots', 'tactic'))
    parser.add_argument('--ball_subdir', default=os.path.join('annots', 'ball'))
    parser.add_argument('--debug_show', type=int, default=20)

    args = parser.parse_args()

    split_info = {
        'train': f'{args.benchmark}_train_split_0.3ratio_info.pkl',
        'val':   f'{args.benchmark}_test_split_0.3ratio_info.pkl'
    }

    out_path_base = os.path.join(os.path.abspath(args.out_folder), args.benchmark)
    out_path = os.path.join(out_path_base, f"C{args.channels}")
    os.makedirs(out_path, exist_ok=True)

    shared_action_map = os.path.join(out_path_base, 'action_map.pkl')

    for part in ['train', 'val']:
        split_file = os.path.join(args.data_path, split_info[part])
        gendata_for_stgcn(
            data_root=args.data_path,
            out_path=out_path,
            split_file_path=split_file,
            part=part,
            channels=args.channels,
            action_map_file=shared_action_map,
            joints_subdir=args.joints_subdir,
            tactic_subdir=args.tactic_subdir,
            ball_subdir=args.ball_subdir,
            debug_show=args.debug_show
        )

    print("\nDone.")
