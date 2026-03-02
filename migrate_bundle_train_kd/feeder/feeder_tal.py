import os
import pickle as pkl
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .feeder_sga import (
    _coerce_action_name,
    _force_to_17_joints,
    _load_array_fix_shape,
    _load_label_map,
    _possible_joint_paths,
    _resolve_split_pkl,
    _read_split_entries,
)


def _safe_segments(action_obj) -> List[Tuple[str, float, float]]:
    segments: List[Tuple[str, float, float]] = []

    if isinstance(action_obj, str):
        segments.append((_coerce_action_name(action_obj), 0.0, 1.0))
        return segments

    if not isinstance(action_obj, dict):
        return segments

    for cls_name, spans in action_obj.items():
        cls_name = _coerce_action_name(cls_name)
        if not isinstance(spans, (list, tuple)):
            continue
        for span in spans:
            if not isinstance(span, (list, tuple)) or len(span) < 2:
                continue
            start = float(span[0])
            end = float(span[1])
            if end > start:
                segments.append((cls_name, start, end))
    return segments


class Feeder(Dataset):
    def __init__(
        self,
        data_root: str,
        split_pkl: str | None = None,
        label_pkl: str | None = None,
        fixed_T: int = 160,
        fps: float = 20.0,
        max_segments: int = 16,
        expected_num_class: int | None = None,
        use_c4: bool = True,
        debug: bool = False,
        **kwargs,
    ):
        self.data_root = data_root
        self.split_pkl = _resolve_split_pkl(data_root, split_pkl, **kwargs)
        self.fixed_T = int(fixed_T)
        self.fps = float(fps)
        self.max_segments = int(max_segments)
        self.use_c4 = bool(use_c4)
        self.debug = bool(debug)

        self.name2id = _load_label_map(data_root, label_pkl)
        self.id2name = [None] * len(self.name2id)
        for k, v in self.name2id.items():
            self.id2name[v] = k

        raw_entries = _read_split_entries(self.split_pkl)
        self.sample_ids: List[str] = []
        for sid, _ in raw_entries:
            if any(os.path.exists(p) for p in _possible_joint_paths(self.data_root, sid)):
                self.sample_ids.append(sid)

        if expected_num_class is not None and len(self.id2name) != int(expected_num_class):
            raise ValueError(f"类别数不符: {len(self.id2name)} vs {expected_num_class}")

        print(f"[FeederTAL] 样本数: {len(self.sample_ids)}, 类别数: {len(self.id2name)}")

    def __len__(self):
        return len(self.sample_ids)

    def _joints_path(self, sid: str) -> str:
        for p in _possible_joint_paths(self.data_root, sid):
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"缺少 joints 文件: {sid}")

    def _tactic_path(self, sid: str) -> str:
        path = os.path.join(self.data_root, "annots", "tactic", f"{sid}_tactic.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    def _ball_path(self, sid: str) -> str:
        return os.path.join(self.data_root, "annots", "ball", f"{sid}_ball_traj.pkl")

    def _load_segments(self, sid: str, orig_t: int, crop_offset: int) -> Tuple[np.ndarray, np.ndarray]:
        class_targets = np.zeros((len(self.id2name), self.fixed_T), dtype=np.float32)
        actionness = np.zeros((self.fixed_T,), dtype=np.float32)

        path = self._tactic_path(sid)
        with open(path, "rb") as f:
            meta = pkl.load(f)

        segs = _safe_segments(meta.get("Action"))
        if not segs:
            return class_targets, actionness

        for cls_name, start_sec, end_sec in segs:
            if cls_name not in self.name2id:
                continue

            start_idx = int(round(start_sec * self.fps)) - crop_offset
            end_idx = int(round(end_sec * self.fps)) - crop_offset

            start_idx = max(0, min(self.fixed_T - 1, start_idx))
            end_idx = max(start_idx + 1, min(self.fixed_T, end_idx))

            cls_id = self.name2id[cls_name]
            class_targets[cls_id, start_idx:end_idx] = 1.0
            actionness[start_idx:end_idx] = 1.0

        return class_targets, actionness

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        obj = np.load(self._joints_path(sid), allow_pickle=True)
        player_keys = None
        if isinstance(obj, np.ndarray) and obj.ndim == 0 and obj.dtype == object:
            data_dict = obj.item()
            if isinstance(data_dict, dict):
                player_keys = list(data_dict.keys())

        x = _load_array_fix_shape(obj)
        x = _force_to_17_joints(x)

        c, orig_t, v, m = x.shape
        crop_offset = 0
        if self.fixed_T and orig_t != self.fixed_T:
            if orig_t > self.fixed_T:
                crop_offset = (orig_t - self.fixed_T) // 2
                x = x[:, crop_offset:crop_offset + self.fixed_T]
            else:
                pad = self.fixed_T - orig_t
                x = np.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)), mode="edge")

        c, t, v, m = x.shape
        if c == 3:
            x = np.concatenate([x, np.zeros((1, t, v, m), dtype=np.float32)], axis=0)
        elif c > 4:
            x = x[:4].astype(np.float32, copy=False)
        elif c < 4:
            x = np.concatenate([x.astype(np.float32), np.zeros((4 - c, t, v, m), dtype=np.float32)], axis=0)
        else:
            x = x.astype(np.float32, copy=False)

        x[3].fill(0.0)
        try:
            with open(self._ball_path(sid), "rb") as f:
                ball_meta = pkl.load(f)
            if isinstance(ball_meta, dict) and player_keys is not None:
                for person_idx, key in enumerate(player_keys[:m]):
                    if key in ball_meta:
                        x[3, :, 11, person_idx] = 1.0
        except Exception:
            pass

        class_targets, actionness = self._load_segments(sid, orig_t=orig_t, crop_offset=crop_offset)

        if not self.use_c4:
            x = x[:3]

        target = {
            "class_targets": torch.from_numpy(class_targets),
            "actionness": torch.from_numpy(actionness),
            "valid_frames": torch.tensor(min(orig_t, self.fixed_T), dtype=torch.long),
        }

        if self.debug:
            print("[FeederTAL DEBUG]", sid, x.shape, class_targets.sum(), actionness.sum())

        return x.astype(np.float32), target, sid
