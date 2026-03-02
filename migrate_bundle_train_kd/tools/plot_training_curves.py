import os, argparse, csv
import matplotlib.pyplot as plt

def load_metrics(csv_path):
    rows = []
    with open(csv_path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            row['epoch'] = int(row['epoch'])
            row['loss'] = float(row.get('loss', 0.0))
            if 'top1' in row: row['top1'] = float(row['top1'])
            if 'top5' in row: row['top5'] = float(row['top5'])
            rows.append(row)
    return rows

def pivot(rows, split, key):
    # 返回 (epochs, values)
    xs, ys = [], []
    for r in rows:
        if r['split'] == split and key in r:
            xs.append(r['epoch'])
            ys.append(float(r[key]))
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', required=True, help='训练输出目录（包含 metrics.csv）')
    ap.add_argument('--out', default=None, help='保存的图片路径（默认保存在 work_dir 下）')
    args = ap.parse_args()

    csv_path = os.path.join(args.work_dir, 'metrics.csv')
    rows = load_metrics(csv_path)

    fig1 = plt.figure()
    x_tr, y_tr = pivot(rows, 'train', 'loss')
    x_va, y_va = pivot(rows, 'val', 'loss')
    plt.plot(x_tr, y_tr, label='train loss')
    plt.plot(x_va, y_va, label='val loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.title('Loss Curves')
    out1 = args.out or os.path.join(args.work_dir, 'curve_loss.png')
    fig1.savefig(out1, dpi=180, bbox_inches='tight')

    if any('top1' in r for r in rows):
        fig2 = plt.figure()
        x, y = pivot(rows, 'val', 'top1')
        plt.plot(x, y, label='val Top1')
        if any('top5' in r for r in rows):
            x5, y5 = pivot(rows, 'val', 'top5')
            plt.plot(x5, y5, label='val Top5')
        plt.xlabel('epoch'); plt.ylabel('accuracy (%)'); plt.legend(); plt.title('Validation Accuracy')
        out2 = os.path.join(args.work_dir, 'curve_acc.png')
        fig2.savefig(out2, dpi=180, bbox_inches='tight')

    print('Saved:', out1, os.path.join(args.work_dir, 'curve_acc.png'))

if __name__ == '__main__':
    main()
