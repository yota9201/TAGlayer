# tools/compute_model_stats.py
import os, sys, argparse, importlib, yaml, torch
from thop import profile, clever_format

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

def import_obj(path):
    m, n = path.rsplit('.', 1)
    mod = __import__(m, fromlist=[n])
    return getattr(mod, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='YAML path 使用训练同一份配置')
    ap.add_argument('--weights', default=None, help='权重文件（可选，不影响FLOPs，仅用于加载以免某些lazy层形状不确定）')
    ap.add_argument('--batch', type=int, default=1, help='统计时的 batch 大小（默认1）')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, 'r'))

    # 取模型/数据关键参数
    mname = cfg['model']
    margs = dict(cfg.get('model_args', {}))
    C = int(margs.get('in_channels', 4))
    T = int(cfg.get('train_feeder_args', {}).get('window_size', 400))
    V = int(margs.get('num_point', 17))
    M = int(margs.get('num_person', 6))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 构建模型
    Model = import_obj(mname)
    model = Model(**margs).to(device).eval()

    # 可选：加载权重（不影响统计，主要防止某些lazy模块报shape错）
    if args.weights:
        sd = torch.load(args.weights, map_location='cpu')
        sd = sd.get('state_dict', sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[load] weights={os.path.basename(args.weights)} | missing={len(missing)} unexpected={len(unexpected)}')

    # 统计参数量
    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Params: total={tot/1e6:.3f}M | trainable={trainable/1e6:.3f}M')

    # 构造 dummy（N,C,T,V,M）
    x = torch.zeros(args.batch, C, T, V, M, device=device)

    # THOP：注意 ST-GCN 接受 5D 输入；返回 MACs（乘加次数），FLOPs≈2*MACs（对多数算子）
    macs, params = profile(model, inputs=(x,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    flops = macs * 2
    flops_str, _ = clever_format([flops, params], "%.3f")

    print(f'MACs:  {macs_str}')
    print(f'FLOPs: {flops_str}   (≈ 2 × MACs)')
    print(f'Params(thop): {params_str}')

if __name__ == '__main__':
    main()
