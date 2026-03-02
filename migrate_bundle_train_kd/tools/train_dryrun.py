# tools/train_dryrun.py
import os
import sys
import importlib
import yaml
import torch
import argparse
from torch.utils.data import DataLoader

# 允许从仓库根目录导包
ROOT = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path.insert(0, ROOT)

def import_obj(path: str):
    mod_path, _, attr = path.rpartition('.')
    if not mod_path:
        return importlib.import_module(path)
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)

# ---------------- argparse ----------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config',
    required=True,
    help='path to yaml config'
)
parser.add_argument(
    '--ckpt',
    default=None,
    help='optional teacher checkpoint'
)
args = parser.parse_args()

CFG = args.config
CKPT = args.ckpt

print('[dryrun] use CFG =', CFG)

# ---------------- load config ----------------
with open(CFG, 'r') as f:
    cfg = yaml.safe_load(f)

print('[dryrun] feeder =', cfg.get('feeder'))
print('[dryrun] train_feeder_args keys =',
      list(cfg.get('train_feeder_args', {}).keys()))

# ---------------- DataLoader ----------------
FeederClass = import_obj(cfg['feeder'])
print('[dryrun] FeederClass =', FeederClass,
      'from', getattr(FeederClass, '__module__', None))

fargs = dict(cfg['train_feeder_args'])

# 关闭随机增强，保证稳态
fargs.pop('random_choose', None)
fargs.pop('random_move', None)

ds = FeederClass(**fargs)
dl = DataLoader(
    ds,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)
import collections
cnt = collections.Counter()
for i, b in zip(range(30), dl):  # 看前30个batch
    lb = b[1] if len(b)==2 else b[1]
    cnt.update(lb.tolist())
print('[label histogram top10]', cnt.most_common(10))


batch = next(iter(dl))
if isinstance(batch, (list, tuple)):
    if len(batch) == 3:
        data, label, _ = batch
    elif len(batch) == 2:
        data, label = batch
    else:
        raise RuntimeError(f'unexpected feeder output len={len(batch)}')
else:
    raise RuntimeError('feeder should return (data, label[, name])')

label = label.long()
print('[label unique]', torch.unique(label).tolist()[:20])
print('[batch] data shape =', tuple(data.shape),
      'label min/max =', int(label.min()), int(label.max()))

# ---------------- Teacher / Student ----------------
TeacherClass = import_obj('net.st_gcn.Model')

Teacher = TeacherClass(
    in_channels=cfg['model_args']['in_channels'],
    num_class=cfg['model_args']['num_class'],
    graph_args=cfg['model_args']['graph_args'],
    edge_importance_weighting=True
).eval()

if CKPT is not None:
    print('[dryrun] loading teacher ckpt:', CKPT)
    sd = torch.load(CKPT, map_location='cpu')
    Teacher.load_state_dict(sd.get('state_dict', sd), strict=False)
else:
    print('[dryrun] no teacher ckpt loaded')

StudentClass = import_obj(cfg['model'])
Student = StudentClass(**cfg['model_args'])
Student.train(True)

bb_flag = getattr(Student.backbone.m, 'training', None) \
    if hasattr(Student, 'backbone') else None
print('[debug] model.training =', Student.training,
      '| backbone.m.training =', bb_flag)

# ---------------- forward & loss ----------------
ce = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    t = Teacher(data)
    s = Student(data)
    loss_t = ce(t, label).item()
    loss_s = ce(s, label).item()

print('Teacher logits stats:',
      'min', float(t.min()),
      'max', float(t.max()),
      'norm', float(t.norm()))
print('Student logits stats:',
      'min', float(s.min()),
      'max', float(s.max()),
      'norm', float(s.norm()))
print('loss_t', loss_t, 'loss_s', loss_s)
print('mean |t-s| =', float((t - s).abs().mean()))

assert label.dtype == torch.long
assert 0 <= int(label.min()) < cfg['model_args']['num_class']
print('[OK] label in [0, num_class-1]')
