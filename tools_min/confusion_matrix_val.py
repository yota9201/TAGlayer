# tools/confusion_matrix_val.py
import os, sys, json, argparse, pickle, numpy as np, torch, yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

def import_obj(path: str):
    m, n = path.rsplit('.', 1)
    mod = __import__(m, fromlist=[n])
    return getattr(mod, n)

def safe_import_feeder(path_str: str):
    # 兼容 'feeder.feeder.Feeder' / 'feeder.feeder'
    if path_str.endswith('.Feeder'):
        return import_obj(path_str)
    mod = __import__(path_str, fromlist=['Feeder'])
    return getattr(mod, 'Feeder') if hasattr(mod, 'Feeder') else import_obj(path_str + '.Feeder')

def load_action_names(path: str|None) -> list|None:
    """
    兼容几种常见格式：

    1) {name: id}  ->  反转得到 index->name
    2) {name: ...list/tuple/...} (SGA-INTERACT 的 GAR_action_info.pkl)
       -> 按 dict 的插入顺序直接用 key 列表作为类别顺序
    3) (names, labels) 形式 -> 直接返回第一个元素
    """
    if not path or not os.path.exists(path):
        return None

    try:
        obj = pickle.load(open(path, 'rb'))

        # case 1: {name: id}
        if isinstance(obj, dict):
            vals = list(obj.values())
            # 全是 int 的话，认为是 name->id 映射
            if len(vals) > 0 and all(isinstance(v, int) for v in vals):
                inv = {v: k for k, v in obj.items()}
                max_id = max(inv) if inv else -1
                return [inv.get(i, f'cls{i:02d}') for i in range(max_id+1)]

            # case 2: SGA-INTERACT 风格 {action_name: list(...)}
            # 我们假设训练时也是按这个顺序把 action 映射到 id 的
            # （python3 中 dict 保序，因此 pickle 读出来也保持原顺序）
            if len(obj) > 0 and any(isinstance(v, (list, tuple)) for v in vals):
                names = list(obj.keys())
                print(f"[info] Detected SGA-style action map, num_classes={len(names)}")
                return names

        # case 3: (names, labels) / [names, labels]
        if isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(obj[0], (list, tuple)):
            return [str(s) for s in obj[0]]

    except Exception as e:
        print(f'[WARN] Failed to read action_map ({e}).')

    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='训练同款 YAML')
    ap.add_argument('--weights', required=True, help='模型权重，如 best_model.pt')
    ap.add_argument('--outdir', default='./eval_out', help='输出目录')
    ap.add_argument('--action-map', default=None, help='覆盖 YAML 中的 action_map 路径')
    ap.add_argument('--annot-as-percent', action='store_true', help='在图中以百分比标注')
    ap.add_argument('--digits', type=int, default=1,
                help='百分比小数位（--annot-as-percent 时生效）')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config, 'r'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) dataloader（val/test）
    Feeder = safe_import_feeder(cfg['feeder'])
    fargs = dict(cfg.get('test_feeder_args', {}))
    ds = Feeder(**fargs)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=int(cfg.get('test_batch_size', 8)),
        shuffle=False, num_workers=int(cfg.get('num_worker', 4)),
        pin_memory=True
    )

    # 2) model
    Model = import_obj(cfg['model'])
    margs = dict(cfg.get('model_args', {}))
    model = Model(**margs).to(device).eval()
    sd = torch.load(args.weights, map_location='cpu')
    sd = sd.get('state_dict', sd)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f'[load] missing={len(miss)} unexpected={len(unexp)}')

    # 3) 推理
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in dl:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, label = batch[0], batch[1]
            else:
                raise ValueError('Unexpected batch format from feeder.')
            data = data.float().to(device, non_blocking=True)
            out = model(data)
            # 兼容 (logits, ...)/dict/tensor
            if isinstance(out, (list, tuple)):
                logits = out[0]
            elif isinstance(out, dict):
                logits = out.get('logits', next(iter(out.values())))
            else:
                logits = out
            pred = logits.argmax(1).cpu().numpy()
            all_pred.append(pred)
            all_true.append(label.numpy())

    y_pred = np.concatenate(all_pred, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    # 4) 类别名
    amp = (args.action_map
           or cfg.get('dataset', {}).get('action_map_path')
           or cfg.get('action_map_path')
           or cfg.get('action_map'))
    class_names = load_action_names(amp)
    if class_names is None:
        # fallback
        num_class = int(cfg.get('model_args', {}).get('num_class', int(np.max(y_true))+1))
        class_names = [f'cls{i:02d}' for i in range(num_class)]
    num_class = len(class_names)

    # 5) 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_class)))
    acc = (y_true == y_pred).mean()
    print(f'[VAL] acc={acc*100:.2f}% | N={len(y_true)} | num_class={num_class}')

    # 行归一化
    cm = cm.astype(np.float64)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / (row_sum + 1e-9)
    print('[CHECK] row sums:', np.round(cm_norm.sum(1), 6))

    # ----------------------------
    # 6) 全 20 类混淆矩阵（原图）
    # ----------------------------
    plt.figure(figsize=(10, 8))
    im = plt.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
    plt.title('Confusion Matrix (Val, row-normalized)')
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Proportion', rotation=270, labelpad=12)
    plt.xticks(np.arange(num_class), class_names, rotation=60, ha='right')
    plt.yticks(np.arange(num_class), class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    thresh = 0.5
    for i in range(num_class):
        for j in range(num_class):
            val = float(cm_norm[i, j])
            if not np.isfinite(val):
                val = 0.0
            if args.annot_as_percent:
                txt = f'{val:.{args.digits}%}'
            else:
                txt = f'{val:.{args.digits}f}'
            plt.text(
                j, i, txt,
                ha='center', va='center',
                color='white' if val > thresh else 'black',
                fontsize=7
            )

    plt.tight_layout()
    full_png = os.path.join(args.outdir, 'confusion_matrix_full.png')
    plt.savefig(full_png, dpi=300, bbox_inches='tight')
    plt.close()
    print('[SAVE] full matrix ->', full_png)

    # ----------------------------
    # 7) 截取大矩阵的左上角 3×3（完全数值一致）
    # ----------------------------
    sub_rows = [0, 1, 2]   # HO, Reverse, PnR 的索引
    sub_cm_norm = cm_norm[np.ix_(sub_rows, sub_rows)]
    sub_names = [class_names[i] for i in sub_rows]

    fig, ax = plt.subplots(figsize=(6, 5))
    im2 = ax.imshow(sub_cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)

    k = len(sub_rows)
    for i in range(k):
        for j in range(k):
            v = float(sub_cm_norm[i, j])
            ax.text(
                j, i, f'{v*100:.0f}%', 
                ha='center', va='center',
                fontsize=18,
                color='white' if v > 0.5 else 'black'
            )

    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(sub_names, rotation=45, ha='right', fontsize=18)
    ax.set_yticklabels(sub_names, fontsize=18)
    ax.set_xlabel('Predicted', fontsize=18)
    ax.set_ylabel('True', fontsize=18)
    ax.set_title('Top-Left 3×3 Sub Confusion Matrix', fontsize=20)

    plt.tight_layout()
    core_png = os.path.join(args.outdir, 'confusion_matrix_top3.png')
    plt.savefig(core_png, dpi=400, bbox_inches='tight', transparent=True)
    plt.close()
    print('[SAVE] small 3×3 matrix ->', core_png)



    # ----------------------------
    # 8) CSV（带类名，仍用全矩阵）
    # ----------------------------
    import csv
    csv_path = os.path.join(args.outdir, 'confusion_matrix.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([''] + class_names)
        for i in range(num_class):
            w.writerow([class_names[i]] + list(map(int, cm[i])))
    print('[SAVE] CSV ->', csv_path)

if __name__ == '__main__':
    main()
