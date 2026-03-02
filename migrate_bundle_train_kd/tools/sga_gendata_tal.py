#!/usr/bin/env python
import argparse
import glob
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

FRAME_RATE = 50
MAX_BODY = 6
NUM_JOINT = 17


def _parse_split_file(split_file_path: str) -> List[str]:
    with open(split_file_path, "rb") as f:
        obj = pickle.load(f)

    clips: List[str] = []
    if isinstance(obj, dict):
        for _cls, items in obj.items():
            for it in items:
                sid = it[0] if isinstance(it, (list, tuple)) else it
                sid = str(sid)
                if sid.endswith("_tactic.pkl"):
                    sid = sid[:-11]
                clips.append(sid)
    elif isinstance(obj, (list, tuple)):
        for sid in obj:
            sid = str(sid)
            if sid.endswith("_tactic.pkl"):
                sid = sid[:-11]
            clips.append(sid)
    else:
        raise ValueError(f"Unsupported split structure: {type(obj)} @ {split_file_path}")
    return clips


def _resolve_joints_path(joints_dir: str, base: str) -> str | None:
    candidates = [
        os.path.join(joints_dir, f"{base}_pose.npy"),
        os.path.join(joints_dir, f"{base}_joints.npy"),
        os.path.join(joints_dir, f"{base}_skeleton.npy"),
        os.path.join(joints_dir, f"{base}.npy"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    hits: List[str] = []
    for patt in [f"{base}*pose*.npy", f"{base}*.npy"]:
        hits.extend(glob.glob(os.path.join(joints_dir, patt)))
    hits = sorted(h for h in hits if os.path.isfile(h))
    return hits[0] if hits else None


def _load_joints_any(path: str) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)

    if isinstance(obj, np.ndarray) and obj.ndim == 0 and obj.dtype == object:
        item = obj.item()
        if isinstance(item, dict):
            out = {}
            for k, v in item.items():
                arr = np.asarray(v)
                if arr.ndim == 3 and arr.shape[2] >= 3:
                    out[str(k)] = arr[:, :NUM_JOINT, :3].astype(np.float32)
            if out:
                return out

    arr = np.asarray(obj)
    if arr.ndim == 4:
        if arr.shape[-1] <= MAX_BODY:
            t, v, c, m = arr.shape
            out = {}
            for idx in range(m):
                out[f"player_{idx + 1}"] = arr[:, :NUM_JOINT, :3, idx].astype(np.float32)
            return out
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return {"player_1": arr[:, :NUM_JOINT, :3].astype(np.float32)}
    return {}


def _normalize_coords_(sample: np.ndarray) -> np.ndarray:
    coords = sample[:3]
    max_val = float(np.abs(coords).max())
    if max_val > 1e-6:
        sample[:3] = coords / max_val
    return sample


def _infer_offense_from_ball(sample: np.ndarray, topk: int = 3) -> np.ndarray:
    ball_sum = sample[3, :, 0, :].sum(axis=0)
    idx_sorted = np.argsort(-ball_sum)
    off_idx = idx_sorted[: min(topk, sample.shape[-1])]
    off_mask = np.zeros((sample.shape[-1],), dtype=np.float32)
    off_mask[off_idx] = 1.0
    return off_mask


def _load_ball_dict(ball_dir: str, base: str) -> Dict:
    path = os.path.join(ball_dir, f"{base}_ball_traj.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        try:
            obj = pickle.load(f)
        except Exception:
            return {}
    return obj if isinstance(obj, dict) else {}


def _clip_segments(action_obj, action_to_id: Dict[str, int], max_frame: int, fps: int) -> Tuple[List[List[int]], int]:
    segments: List[List[int]] = []
    dropped = 0

    if not isinstance(action_obj, dict):
        return segments, dropped

    for action_name, spans in action_obj.items():
        if action_name not in action_to_id:
            continue
        if not isinstance(spans, (list, tuple)):
            continue
        cls = int(action_to_id[action_name])
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) < 2:
                dropped += 1
                continue
            st = int(float(span[0]) * fps)
            ed = int(float(span[1]) * fps)
            st = max(0, min(max_frame, st))
            ed = max(0, min(max_frame, ed))
            if ed <= st:
                dropped += 1
                continue
            segments.append([st, ed, cls])
    return segments, dropped


def gendata_tal(
    data_root: str,
    split_file_path: str,
    out_dir: str,
    part: str,
    channels: int,
    max_frame: int,
    fps: int,
    action_map_file: str,
):
    joints_dir = os.path.join(data_root, "joints")
    tactic_dir = os.path.join(data_root, "annots", "tactic")
    ball_dir = os.path.join(data_root, "annots", "ball")

    clip_ids = _parse_split_file(split_file_path)
    action_to_id: Dict[str, int] = {}
    if os.path.exists(action_map_file):
        with open(action_map_file, "rb") as f:
            action_to_id = pickle.load(f)

    samples = []
    names = []
    segments_all = []

    stats = {
        "missing_joints": 0,
        "bad_joints": 0,
        "missing_tactic": 0,
        "bad_tactic": 0,
        "empty_segments": 0,
        "dropped_segments": 0,
    }

    for sid in clip_ids:
        joints_path = _resolve_joints_path(joints_dir, sid)
        if joints_path is None:
            stats["missing_joints"] += 1
            continue

        joints_dict = _load_joints_any(joints_path)
        if not joints_dict:
            stats["bad_joints"] += 1
            continue

        tactic_path = os.path.join(tactic_dir, f"{sid}_tactic.pkl")
        if not os.path.exists(tactic_path):
            stats["missing_tactic"] += 1
            continue

        with open(tactic_path, "rb") as f:
            try:
                tac = pickle.load(f)
            except Exception:
                stats["bad_tactic"] += 1
                continue

        action_obj = tac.get("Action", {})
        if part == "train":
            if isinstance(action_obj, dict):
                for action_name in action_obj.keys():
                    if action_name not in action_to_id:
                        action_to_id[action_name] = len(action_to_id)
        elif not action_to_id:
            raise FileNotFoundError(f"Missing action_map.pkl for val: {action_map_file}")

        sample = np.zeros((channels, max_frame, NUM_JOINT, MAX_BODY), dtype=np.float32)
        player_keys = list(joints_dict.keys())[:MAX_BODY]

        valid_len = 0
        for m, key in enumerate(player_keys):
            arr = np.asarray(joints_dict[key], dtype=np.float32)
            if arr.ndim != 3:
                continue
            t_use = min(arr.shape[0], max_frame)
            v_use = min(arr.shape[1], NUM_JOINT)
            sample[:3, :t_use, :v_use, m] = np.transpose(arr[:t_use, :v_use, :3], (2, 0, 1))
            valid_len = max(valid_len, t_use)

        if valid_len == 0:
            stats["bad_joints"] += 1
            continue

        if channels >= 4:
            ball_dict = _load_ball_dict(ball_dir, sid)
            for m, key in enumerate(player_keys):
                traj = ball_dict.get(key)
                if not hasattr(traj, "__len__"):
                    continue
                t_use = min(len(traj), max_frame)
                sample[3, :t_use, :, m] = 1.0

        if channels == 5:
            off_mask = _infer_offense_from_ball(sample, topk=3)
            for m in range(MAX_BODY):
                sample[4, :, :, m] = off_mask[m]

        sample = _normalize_coords_(sample)

        segments, dropped = _clip_segments(action_obj, action_to_id, max_frame=max_frame, fps=fps)
        stats["dropped_segments"] += dropped
        if not segments:
            stats["empty_segments"] += 1
            continue

        samples.append(sample)
        names.append(sid)
        segments_all.append({"sid": sid, "segments": segments, "length": int(valid_len)})

    os.makedirs(out_dir, exist_ok=True)
    data = np.stack(samples, axis=0) if samples else np.zeros((0, channels, max_frame, NUM_JOINT, MAX_BODY), dtype=np.float32)
    np.save(os.path.join(out_dir, f"{part}_data.npy"), data)
    with open(os.path.join(out_dir, f"{part}_segments.pkl"), "wb") as f:
        pickle.dump(segments_all, f)
    with open(os.path.join(out_dir, f"{part}_names.pkl"), "wb") as f:
        pickle.dump(names, f)
    if part == "train":
        with open(action_map_file, "wb") as f:
            pickle.dump(action_to_id, f)

    avg_segments = float(np.mean([len(x["segments"]) for x in segments_all])) if segments_all else 0.0
    print(f"[{part}] samples={len(samples)} avg_segments={avg_segments:.2f} dropped_segments={stats['dropped_segments']}")
    for k, v in stats.items():
        print(f"[{part}] {k}: {v}")
    for item in segments_all[:3]:
        print(f"[{part}] sanity {item['sid']}: {item['segments'][:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGA-INTERACT TAL gendata")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_folder", required=True)
    parser.add_argument("--benchmark", default="GAL", choices=["GAL", "TGAL"])
    parser.add_argument("--channels", type=int, default=4, choices=[3, 4, 5])
    parser.add_argument("--max_frame", type=int, default=400)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    benchmark = args.benchmark
    out_dir = os.path.join(os.path.abspath(args.out_folder), benchmark, f"C{args.channels}")
    action_map_file = os.path.join(os.path.abspath(args.out_folder), benchmark, "action_map.pkl")

    for part, split_name in [("train", f"{benchmark}_train_split_0.3ratio_info.pkl"), ("val", f"{benchmark}_test_split_0.3ratio_info.pkl")]:
        split_path = os.path.join(args.data_path, split_name)
        gendata_tal(
            data_root=args.data_path,
            split_file_path=split_path,
            out_dir=out_dir,
            part=part,
            channels=args.channels,
            max_frame=args.max_frame,
            fps=args.fps,
            action_map_file=action_map_file,
        )

