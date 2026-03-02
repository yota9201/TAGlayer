# tools/eval_student.py
import os, sys, yaml, importlib, torch
from torch.utils.data import DataLoader

# 把项目根加入 sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))           # .../st-gcn/tools
sys.path.insert(0, os.path.dirname(ROOT))                   # .../st-gcn

CFG = 'config/feature_train/shift_e2e_stgcnbackbone_c4_baseline.yaml'  # 你的基线yaml
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def import_obj(dotted: str):
    """'pkg.mod.Class' -> Class对象"""
    mod, cls = dotted.rsplit('.', 1)
    return getattr(importlib.import_module(mod), cls)

with open(CFG, 'r') as f:
    cfg = yaml.safe_load(f)

# ===== 1) DataLoader（val集；禁用增强）=====
FeederCls = import_obj(cfg['feeder'])   # 这里会把 'feeder.feeder.Feeder' 正确拆开
fargs = dict(cfg['test_feeder_args'])
fargs['random_choose'] = False
fargs['random_move'] = False
ds = FeederCls(**fargs)
dl = DataLoader(ds, batch_size=cfg.get('test_batch_size', 8),
                shuffle=False, num_workers=cfg.get('num_worker', 4), pin_memory=True)

# ===== 2) 模型（端到端学生：冻结骨干、无TAG/TA/语义；会在 __init__ 打印“Init classifier loaded...”）=====
ModelCls = import_obj(cfg['model'])
model = ModelCls(**cfg['model_args']).to(DEVICE).eval()

# ===== 3) Eval loop =====
top1 = top5 = total = 0
with torch.no_grad():
    for batch in dl:
        # 兼容 (data,label) 或 (data,label,name)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            data, label = batch[0], batch[1]
        else:
            raise RuntimeError("Unexpected feeder batch format; expect (data,label, ...)")
        data = data.float().to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)

        logits = model(data)                     # (N, num_class)
        # top-1 / top-5
        top5_pred = logits.topk(5, dim=1).indices  # (N,5)
        total += label.size(0)
        top1 += (top5_pred[:, 0] == label).sum().item()
        top5 += (top5_pred == label.unsqueeze(1)).any(dim=1).sum().item()

print(f'[Eval-Student] Top1: {top1/total*100:.2f}%  Top5: {top5/total*100:.2f}%  N={total}')
