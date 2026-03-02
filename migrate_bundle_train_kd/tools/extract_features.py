# tools/extract_features.py
# -*- coding: utf-8 -*-
"""
从预训练/微调后的 ST-GCN 主干提取 GAP 前特征图，并流式写盘避免 OOM。
输出三件文件（每个 split 一套）：
  - features_{tag}_{split}.npy         # (N, C_feat, T', V)  float16/float32
  - features_{tag}_{split}_labels.npy  # (N, )               int64
  - features_{tag}_{split}_names.pkl   # list[str]           样本名
  - features_{tag}_{split}_meta.json   # 元信息
"""

import os
import sys
import json
import pickle
import importlib
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

# 让仓库根目录入 sys.path（无论从哪里运行）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ----------------- 工具函数 -----------------
def import_obj(path: str):
    """按 'pkg.mod.Class' 导入对象"""
    m, n = path.rsplit('.', 1)
    return getattr(importlib.import_module(m), n)

def load_yaml(p):
    import yaml
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_weights_with_ignore(model, ckpt_path, ignore_list=None):
    """非严格加载；支持忽略前缀/精确键；自动展开 state_dict 包装。"""
    if not ckpt_path:
        print('[INFO] 未提供 weights，跳过加载')
        return model

    print(f'[INFO] 加载权重: {ckpt_path}')
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict):
        for k in ['state_dict', 'model', 'model_state', 'weights']:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    sd = dict(state)

    ig = set(ignore_list or [])
    if ig:
        drop = []
        for k in list(sd.keys()):
            if (k in ig) or any(k.startswith(p) for p in ig):
                drop.append(k)
        for k in drop:
            sd.pop(k, None)
        print(f'[INFO] 忽略 {len(drop)} 项权重（来自 ignore_weights）')

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[INFO] missing={len(missing)}, unexpected={len(unexpected)}')
    return model

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# ----------------- 主逻辑 -----------------
@torch.no_grad()
def extract_split(cfg: dict,
                  split: str,
                  out_dir: str,
                  tag: str,
                  device: str = 'cuda',
                  person_pool: str = 'max',   # {'max','mean'}
                  t_downsample: int = 1,      # 时间下采样步长（>=1）
                  dtype: str = 'float16',     # {'float16','float32'}
                  batch_size_override: int | None = None,
                  num_worker_override: int | None = None):
    """
    从指定 split（train/val）抽特征并写盘。
    """
    assert split in ('train', 'val', 'test')
    ensure_dir(out_dir)

    # 1) Feeder & DataLoader
    Feeder = import_obj(cfg['feeder'])
    fkey = 'train_feeder_args' if split == 'train' else 'test_feeder_args'
    fargs = dict(cfg.get(fkey, {}))

    # 抽特征阶段禁用随机增强
    if 'random_choose' in fargs: fargs['random_choose'] = False
    if 'random_move'   in fargs: fargs['random_move']   = False

    ds = Feeder(**fargs)
    bs_cfg = cfg.get('test_batch_size', 8)
    nw_cfg = cfg.get('num_worker', 4)
    bs = batch_size_override if batch_size_override is not None else bs_cfg
    nw = num_worker_override if num_worker_override is not None else nw_cfg

    dl = DataLoader(
        ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True, drop_last=False
    )

    # 2) Model（eval/frozen）
    Model = import_obj(cfg['model'])
    model = Model(**cfg.get('model_args', {})).to(device).eval()
    model = load_weights_with_ignore(model, cfg.get('weights'), cfg.get('ignore_weights'))

    # 3) Hook：捕获最后一层（GAP 前）的特征图
    buf = []
    def hook(_m, _i, o):
        x = o[0] if isinstance(o, (list, tuple)) else o
        # 注意：有些实现此处输出为 (N*M, C, T, V)，不能在 hook 里聚合 M（拿不到 batch 的 M）
        # 我们只做时间下采样；人员聚合放到主循环里，依据当批 data 的形状复原 M
        if x.dim() == 5:
            # 罕见：有实现输出未拍平人维，这时可先聚合到 (N, C, T, V)
            x = x.max(dim=-1).values if person_pool == 'max' else x.mean(dim=-1)
        if t_downsample and t_downsample > 1:
            x = x[:, :, ::t_downsample, :]
        buf.append(x.detach().cpu())

    target = None
    if hasattr(model, 'st_gcn_networks') and len(model.st_gcn_networks) > 0:
        target = model.st_gcn_networks[-1]
    elif hasattr(model, 'l10'):  # 一些实现最后块叫 l10
        target = model.l10
    else:
        print('[WARN] 未找到 st_gcn_networks[-1]/l10，将打印前 50 个模块名供参考：')
        print(list(dict(model.named_modules()).keys())[:50])
        raise RuntimeError('未找到最后一层模块，请修改 extract_features.py 中的钩子定位逻辑')

    h = target.register_forward_hook(hook)

    # 4) 计算总样本数并准备流式写盘
    N_total = len(ds)
    print(f'[INFO] Split={split}，样本数 N={N_total}，batch_size={bs}，num_workers={nw}')

    base = os.path.join(out_dir, f'features_{tag}_{split}')
    # 先准备标签 memmap（维度可提前确定）
    labels_mm = np.lib.format.open_memmap(base + '_labels.npy', mode='w+', dtype=np.int64, shape=(N_total,))
    names_all = []   # 名字占内存极小，用 list 累积

    feat_mm = None   # 第一个 batch 拿到特征形状后再分配
    offset = 0
    use_fp16 = (dtype == 'float16')

    # 5) 遍历 DataLoader（流式写盘）
    name_list = getattr(ds, 'sample_name', None)

    for bi, batch in enumerate(dl):
        # —— 兼容不同 Feeder 的返回 —— #
        if len(batch) == 3:
            data, label, idx_or_name = batch
            # 解析名字
            if name_list is not None:
                if torch.is_tensor(idx_or_name):
                    cur_names = [str(name_list[int(i)]) for i in idx_or_name]
                elif isinstance(idx_or_name, (list, tuple)):
                    cur_names = [str(name_list[int(i)]) for i in idx_or_name]
                elif isinstance(idx_or_name, (int, np.integer)):
                    cur_names = [str(name_list[int(idx_or_name)])]
                else:
                    cur_names = [str(idx_or_name)]
            else:
                if torch.is_tensor(idx_or_name):
                    cur_names = [str(int(i)) for i in idx_or_name]
                elif isinstance(idx_or_name, (list, tuple)):
                    cur_names = [str(int(i)) for i in idx_or_name]
                else:
                    cur_names = [str(idx_or_name)]
        else:
            data, label = batch
            # 占位名
            bs_input = data.shape[0]
            cur_names = [f"{split}_{offset + k}" for k in range(bs_input)]

        # 前传（触发 hook）
        data = data.float().to(device, non_blocking=True)
        _ = model(data)

        # 取出本 batch 的特征
        x = buf.pop(0)  # 可能是 (N*C,...) 也可能是 (N,...)，取决于实现

        # === 关键修正：如果输出拍平了人维 (N*M, C, T, V)，按当前 batch 输入的 M 复原并池化到 (N, C, T, V) ===
        if x.dim() == 4 and data.dim() == 5:
            bs_input = data.shape[0]
            M_input  = data.shape[-1]
            if x.shape[0] == bs_input * M_input:
                C, T, V = x.shape[1], x.shape[2], x.shape[3]
                x = x.view(bs_input, M_input, C, T, V)
                x = x.max(dim=1).values if person_pool == 'max' else x.mean(dim=1)
            # 否则视为已是 (N, C, T, V)

        # 现在 x 必须是 (N, C, T, V)
        bs_cur, C, T, V = x.shape

        # 第一次见到形状，创建特征 memmap
        if feat_mm is None:
            feat_dtype = np.float16 if use_fp16 else np.float32
            feat_mm = np.lib.format.open_memmap(base + '.npy', mode='w+', dtype=feat_dtype, shape=(N_total, C, T, V))
            meta = {
                'C_feat': int(C), 'T_feat': int(T), 'V': int(V),
                'dtype': 'float16' if use_fp16 else 'float32',
                'person_pool': person_pool, 't_downsample': int(t_downsample)
            }
            with open(base + '_meta.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 特征形状: (N={N_total}, C={C}, T={T}, V={V}), dtype={meta['dtype']}")

        # 写入本 batch
        if use_fp16:
            feat_mm[offset:offset+bs_cur] = x.numpy().astype(np.float16, copy=False)
        else:
            feat_mm[offset:offset+bs_cur] = x.numpy().astype(np.float32, copy=False)

        labels_mm[offset:offset+bs_cur] = label.cpu().numpy().astype(np.int64, copy=False)
        names_all.extend(cur_names)
        offset += bs_cur

        if (bi + 1) % 20 == 0:
            print(f'[INFO] {split}: {offset}/{N_total} done')

    h.remove()  # 去掉 hook

    # 刷盘
    del feat_mm
    del labels_mm

    with open(base + '_names.pkl', 'wb') as f:
        pickle.dump(names_all, f)

    print(f'[OK] {split} 完成：')
    print(f'  - {base}.npy')
    print(f'  - {base}_labels.npy')
    print(f'  - {base}_names.pkl')
    print(f'  - {base}_meta.json')

# ----------------- CLI -----------------
def main():
    p = argparse.ArgumentParser(description='Extract feature maps from ST-GCN backbone (streaming, OOM-safe)')
    p.add_argument('-c', '--config', required=True, help='YAML 配置路径（与你训练/微调时一致）')
    p.add_argument('--out_dir', required=True, help='输出目录')
    p.add_argument('--tag', required=True, help='文件名前缀标签，例如 c4_ft')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                   choices=['cuda', 'cpu'])
    p.add_argument('--person_pool', default='max', choices=['max', 'mean'],
                   help='当特征仍带 M 维时的聚合方式')
    p.add_argument('--t_downsample', type=int, default=1,
                   help='时间下采样步长（>=1；2 表示时间维减半）')
    p.add_argument('--dtype', default='float16', choices=['float16', 'float32'],
                   help='特征存盘精度，float16 可将体积减半')
    p.add_argument('--batch_size', type=int, default=None,
                   help='可选：覆盖 YAML 的 test_batch_size')
    p.add_argument('--num_worker', type=int, default=None,
                   help='可选：覆盖 YAML 的 num_worker')
    args = p.parse_args()

    cfg = load_yaml(args.config)
    ensure_dir(args.out_dir)

    for split in ['train', 'val']:
        extract_split(cfg, split,
                      out_dir=args.out_dir,
                      tag=args.tag,
                      device=args.device,
                      person_pool=args.person_pool,
                      t_downsample=max(1, int(args.t_downsample)),
                      dtype=args.dtype,
                      batch_size_override=args.batch_size,
                      num_worker_override=args.num_worker)

if __name__ == '__main__':
    main()
