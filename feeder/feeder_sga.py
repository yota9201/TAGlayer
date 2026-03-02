# feeder/feeder_sga.py
import os
import glob
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Any
print('[FeederSGA] loaded from:', __file__)


# ---------------- 工具函数 ----------------

def _coerce_action_name(name: str) -> str:
    return str(name).strip()


def _load_label_map(data_root: str, label_pkl: str | None = None) -> dict:
    """
    读取动作类别表，返回 name2id 映射。
    兼容：
      - 路径：data_root/<label_pkl> 或 data_root/annots/<label_pkl>
      - 结构：defaultdict / dict 的 {class_name: [...]}
             或 list[(class_name, items)]
             或 dict{'classes': [class_name, ...]}
    """
    if label_pkl is None:
        label_pkl = "GAR_action_info.pkl"

    cand = [
        os.path.join(data_root, label_pkl),
        os.path.join(data_root, "annots", label_pkl),
    ]
    p = next((c for c in cand if os.path.exists(c)), None)
    if p is None:
        print(f"[WARN] 未找到 {label_pkl}，后续可能通过样本扫描补齐类别表。")
        return {}

    with open(p, "rb") as f:
        obj = pkl.load(f)

    if isinstance(obj, dict) and "classes" not in obj:
        classes = [str(k) for k in obj.keys()]
    elif isinstance(obj, list):
        classes = [str(it[0]) for it in obj if isinstance(it, (list, tuple)) and len(it) >= 1]
    elif isinstance(obj, dict) and "classes" in obj and isinstance(obj["classes"], (list, tuple)):
        classes = [str(x) for x in obj["classes"]]
    else:
        raise ValueError(f"{label_pkl} 的结构无法解析: {type(obj)}")

    name2id: dict[str, int] = {}
    for cls in classes:
        n = _coerce_action_name(cls)
        if n not in name2id:
            name2id[n] = len(name2id)

    if not name2id:
        raise ValueError(f"{label_pkl} 中未解析到任何类别。")

    return name2id


def _resolve_split_pkl(data_root: str, split_pkl: str | None, **kwargs) -> str | None:
    """
    支持三种写法：
      1) 直接给文件名: GAR_train_split_0.3ratio_info.pkl
      2) 给完整路径: /path/to/GAR_train_split_0.3ratio_info.pkl
      3) 关键字: 'train' 或 'test'
         自动在 data_root 或 data_root/annots 下查找 GAR_train/test_split_*_info.pkl
    """
    def _coalesce(d, *keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    if split_pkl is None:
        split_pkl = _coalesce(kwargs, "split_path", "split_file", "split")

    if split_pkl is None:
        return None

    if os.path.isabs(split_pkl) and os.path.exists(split_pkl):
        return split_pkl

    cand = [
        os.path.join(data_root, split_pkl),
        os.path.join(data_root, "annots", split_pkl),
    ]
    for c in cand:
        if os.path.exists(c):
            return c

    if split_pkl in {"train", "test"}:
        patts = [
            os.path.join(data_root, f"GAR_{split_pkl}_split_*_info.pkl"),
            os.path.join(data_root, "annots", f"GAR_{split_pkl}_split_*_info.pkl"),
        ]
        hits = []
        for p in patts:
            hits.extend(sorted(glob.glob(p)))
        if not hits:
            raise FileNotFoundError(f"未在 {data_root} 找到 GAR_{split_pkl}_split_*_info.pkl")
        pref = [h for h in hits if "0.3ratio" in os.path.basename(h)]
        return pref[0] if pref else hits[0]

    raise FileNotFoundError(f"未找到 split pkl: {split_pkl}")


def _read_split_entries(path: str) -> List[Tuple[str, str]]:
    """
    从 split pkl 中读取 (sid, cls_name)
    """
    with open(path, "rb") as f:
        obj = pkl.load(f)

    entries: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for cls_name, items in obj.items():
            for it in items:
                sid = it[0] if isinstance(it, (list, tuple)) else it
                sid = str(sid).replace("_tactic.pkl", "").replace(".pkl", "").replace("_pose.npy", "")
                entries.append((sid, cls_name))
    elif isinstance(obj, (list, tuple)):
        for sid in obj:
            sid = str(sid).replace("_tactic.pkl", "").replace(".pkl", "").replace("_pose.npy", "")
            entries.append((sid, None))
    else:
        raise ValueError(f"不支持的 split 结构: {type(obj)} @ {path}")
    return entries


def _from_dict_like(d: dict) -> np.ndarray:
    # d: {pid: ndarray(T,V,C)}，M个球员
    mats = []
    for i, (pid, arr) in enumerate(d.items()):
        arr = np.asarray(arr)  # (T,V,C)
        if arr.ndim != 3:
            raise ValueError(f"意外 joints shape: {arr.shape} @ {pid}")
        T, V, C = arr.shape
        mats.append(arr[..., :3][None, ...])  # (1,T,V,3)
    X = np.concatenate(mats, axis=0)  # (M,T,V,3)
    X = np.transpose(X, (3,1,2,0))    # (C,T,V,M)
    return X.astype(np.float32)


def _load_array_fix_shape(obj) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0 and obj.dtype == object:
            return _from_dict_like(obj.item())
        if obj.ndim == 4:  # (T,M,V,C)
            T,M,V,C = obj.shape
            X = np.transpose(obj, (3,0,2,1))  # -> (C,T,V,M)
            return X.astype(np.float32)
        raise ValueError(f"无法识别或不支持的 joints 维度: {obj.shape}, 期望 3D/4D")
    elif isinstance(obj, dict):
        return _from_dict_like(obj)
    else:
        raise TypeError(f"未知 joints 类型: {type(obj)}")


def _load_joints_file(path: str) -> np.ndarray:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if "data" in obj:
            obj = obj["data"]
        else:
            raise ValueError(f"{path} npz 缺少 data 键")
    return _load_array_fix_shape(obj)


def _possible_joint_paths(data_root: str, sid: str) -> List[str]:
    return [
        os.path.join(data_root, "joints", f"{sid}.npy"),
        os.path.join(data_root, "joints", f"{sid}_pose.npy"),
    ]
# ==== 18关节(OpenPose) → 17关节 兼容补丁 ====
# OpenPose-18 的 idx=1 是 Neck；多数 17 点图不包含它
_OPENPOSE18_TO_17_IDX = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

def _force_to_17_joints(arr: np.ndarray) -> np.ndarray:
    """
    输入 arr 形状为 (C,T,V,M) 或 (C,T,V)，若 V=18 则裁掉 Neck 变为 17 点。
    其他情况原样返回。
    """
    if arr.ndim == 4 and arr.shape[2] == 18:
        return arr[:, :, _OPENPOSE18_TO_17_IDX, :]
    if arr.ndim == 3 and arr.shape[2] == 18:
        return arr[:, :, _OPENPOSE18_TO_17_IDX]
    return arr



# ---------------- Dataset ----------------

class Feeder(Dataset):
    def __init__(
        self,
        data_root: str,
        split_pkl: str | None = None,
        label_pkl: str | None = None,
        fixed_T: int = 160,
        expected_num_class: int | None = None,
        strict: bool = True,
        ignore_orphan_joints: bool = True,
        allow_conflict: bool = True,
        debug_samples: int | None = None,
        **kwargs,
    ):
        self.data_root = data_root
        self.split_pkl = _resolve_split_pkl(data_root, split_pkl, **kwargs)
        self.label_pkl = label_pkl
        self.fixed_T = fixed_T
        self.strict = strict
        self.ignore_orphan_joints = ignore_orphan_joints
        self.allow_conflict = allow_conflict

        # 类别映射
        self.name2id = _load_label_map(data_root, label_pkl)
        self.id2name = [None]*len(self.name2id)
        for k,v in self.name2id.items():
            self.id2name[v] = k

        # split entries
        raw_entries = _read_split_entries(self.split_pkl)
        sample_ids, labels = [], []
        for sid, cls in raw_entries:
            if cls is None:  # 没有类别，用 tactic 兜底
                try:
                    _, cls_name, _ = self._load_tactic(sid)
                    cls = cls_name
                except Exception:
                    continue
            if cls not in self.name2id:
                continue
            jp = _possible_joint_paths(self.data_root, sid)
            chosen = None
            for p in jp:
                if os.path.exists(p):
                    chosen = p
                    break
            if chosen is None:
                continue
            sample_ids.append(sid)
            labels.append(self.name2id[cls])

        if debug_samples:
            sample_ids = sample_ids[:debug_samples]
            labels = labels[:debug_samples]

        self.sample_ids = sample_ids
        self.labels = labels

        if expected_num_class is not None and len(self.id2name) != expected_num_class:
            raise ValueError(f"类别数不符: {len(self.id2name)} vs {expected_num_class}")
        print(f"[FeederSGA] 样本数: {len(self.sample_ids)}, 类别数: {len(self.id2name)}")

    def __len__(self):
        return len(self.sample_ids)

    def _joints_path(self, sid: str) -> str:
        for p in _possible_joint_paths(self.data_root, sid):
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"缺少 joints 文件: {sid}")

    def _load_tactic(self, sid: str) -> Tuple[int,str,Any]:
        path = os.path.join(self.data_root,"annots","tactic",f"{sid}_tactic.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path,"rb") as f:
            d = pkl.load(f)
        if not isinstance(d,dict) or "Action" not in d:
            raise KeyError(f"{sid} tactic.pkl missing 'Action'")
        action_name = _coerce_action_name(d["Action"])
        offensive = d.get("Offensive", None)
        if action_name not in self.name2id:
            if self.strict:
                raise ValueError(f"Class {action_name} 不在类别映射中")
            else:
                self.name2id[action_name] = len(self.name2id)
                self.id2name.append(action_name)
        label = self.name2id[action_name]
        return label, action_name, offensive
    
    def _ball_path(self, sid: str) -> str:
        cand = [
            os.path.join(self.data_root, "annots", "ball", f"{sid}_ball_traj.pkl"),
            os.path.join(self.data_root, "annots", "ball", f"{sid}_ball.pkl"),
            os.path.join(self.data_root, "annots", "ball", f"{sid}_balltraj.pkl"),
        ]
        for p in cand:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"缺少 ball 文件: {sid} (tried {cand})")
        
    def _load_ball_traj(self, sid: str) -> np.ndarray:
        path = self._ball_path(sid)
        with open(path, "rb") as f:
            d = pkl.load(f)

        # 兼容多种结构，最终返回 (T,3) float32
        if isinstance(d, dict):
            for k in ["traj", "ball_traj", "ball", "xyz", "pos", "positions"]:
                if k in d:
                    arr = np.asarray(d[k])
                    break
            else:
                # 有些可能直接 dict['data']
                arr = np.asarray(d.get("data"))
        else:
            arr = np.asarray(d)

        if arr is None:
            raise ValueError(f"ball traj parse failed: {sid} @ {path}")

        arr = arr.astype(np.float32)
        if arr.ndim == 1 and arr.size == 3:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"unexpected ball traj shape: {arr.shape} @ {sid}")
        return arr[:, :3]


    def __getitem__(self, idx):
        sid = self.sample_ids[idx]

        x = _load_joints_file(self._joints_path(sid))  # (C,T,V,M) 预计 C=3
        x = _force_to_17_joints(x)                     # (3,T,17,M)

        # 固定长度对齐
        C, T, V, M = x.shape
        if self.fixed_T and T != self.fixed_T:
            if T > self.fixed_T:
                start = (T - self.fixed_T) // 2
                x = x[:, start:start + self.fixed_T]
            else:
                pad = self.fixed_T - T
                x = np.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)), "edge")

        # 重新取对齐后的形状
        C, T, V, M = x.shape

        # 永远创建 extra 通道（默认全0）
        extra = np.zeros((1, T, V, M), dtype=np.float32)

        # 尝试用 ball 填充 extra（失败也无所谓，至少 C=4 不会丢）
        try:
            ball = self._load_ball_traj(sid)  # (Tb,3)
            Tb = ball.shape[0]

            if Tb >= T:
                start = (Tb - T) // 2
                ball_t = ball[start:start + T]
            else:
                pad = T - Tb
                ball_t = np.pad(ball, ((0, pad), (0, 0)), mode="edge")

            root_idx = 0  # 先保底
            root_xyz = x[0:3, :, root_idx, :]            # (3,T,M)
            root_xyz = np.transpose(root_xyz, (1, 0, 2)) # (T,3,M)
            bt = ball_t[:, :, None]                      # (T,3,1)

            dist = np.linalg.norm(root_xyz - bt, axis=1) # (T,M)
            inv = 1.0 / (dist + 1e-3)                    # 越近越大
            extra[0] = inv[:, None, :].repeat(V, axis=1) # (T,V,M)

        except Exception as e:
            # 可选：你想看 ball 失败原因就打开这行
            # print('[WARN] ball extra failed for', sid, '->', repr(e))
            pass

        # 拼接成 4 通道（关键：无论ball成功与否都拼）
        x = np.concatenate([x.astype(np.float32), extra], axis=0)  # (4,T,V,M)

        label = self.labels[idx]
        # 临时调试：确认是否真的拼到4通道
        if idx == 0:
            print('[FeederSGA DEBUG] sid', sid, 'x.shape before return =', x.shape)

        return x, int(label)

