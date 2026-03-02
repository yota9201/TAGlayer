import os
import re
import json
import argparse
from collections import defaultdict

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def infer_k_from_path(path):
    m = re.search(r'[_\-]k(\d+)', path)
    return int(m.group(1)) if m else None

def infer_epoch_from_filename(fn):
    m = re.search(r'epoch(\d+)', fn)
    return int(m.group(1)) if m else None

def parse_perclass(obj):
    """
    return: names(list[str]), acc(list[float]), support(list[int] or zeros)
    """
    if isinstance(obj, list):
        names, accs, supp = [], [], []
        for it in obj:
            names.append(str(it.get("name", it.get("class_name", it.get("id", len(names))))))
            accs.append(float(it.get("acc", it.get("top1", it.get("accuracy", 0.0)))))
            supp.append(int(it.get("support", it.get("count", it.get("n", 0)))))
        return names, accs, supp

    if isinstance(obj, dict):
        accs = obj.get("per_class_acc") or obj.get("acc") or obj.get("top1")
        names = obj.get("class_names") or obj.get("name") or obj.get("classes")
        supp = obj.get("support") or obj.get("count") or obj.get("n")
        if accs is None:
            # dict keyed by class name
            if all(isinstance(v, dict) for v in obj.values()):
                names = list(obj.keys())
                accs = [float(obj[n].get("acc", obj[n].get("top1", 0.0))) for n in names]
                supp = [int(obj[n].get("support", obj[n].get("count", 0))) for n in names]
            else:
                raise ValueError("Unknown perclass.json format(dict).")
        if names is None:
            names = [str(i) for i in range(len(accs))]
        if supp is None:
            supp = [0 for _ in range(len(accs))]
        return list(map(str, names)), list(map(float, accs)), list(map(int, supp))

    raise ValueError("Unknown perclass.json format.")

def macro_mean(accs):
    return sum(accs) / len(accs) if accs else 0.0

def weighted_mean(accs, supp):
    tot = sum(supp)
    if tot <= 0:
        return macro_mean(accs)
    return sum(a * s for a, s in zip(accs, supp)) / tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="work_dir/recognition/tag 的父目录（递归扫描）")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--metric", default="weighted", choices=["macro", "weighted"],
                    help="选择 best epoch 的指标：macro 或 weighted（默认 weighted 更接近 overall acc）")
    ap.add_argument("--epochs", default="0-39", help="只考虑哪些 epoch，如 0-39 或 0,1,2,10")
    ap.add_argument("--out_json", default="k_sweep_best.json")
    ap.add_argument("--out_tsv", default="k_sweep_best_table.tsv")
    args = ap.parse_args()

    # parse epochs
    ep_set = set()
    if "-" in args.epochs and "," not in args.epochs:
        a, b = args.epochs.split("-")
        ep_set = set(range(int(a), int(b) + 1))
    else:
        ep_set = set(int(x.strip()) for x in args.epochs.split(",") if x.strip())

    # collect perclass files: k -> epoch -> path
    perclass_map = defaultdict(dict)
    for r, ds, fs in os.walk(args.root):
        for fn in fs:
            if not fn.endswith("_perclass.json"):
                continue
            ep = infer_epoch_from_filename(fn)
            if ep is None or ep not in ep_set:
                continue
            p = os.path.join(r, fn)
            k = infer_k_from_path(p)
            if k is None or not (args.kmin <= k <= args.kmax):
                continue
            perclass_map[k][ep] = p

    ks = sorted(perclass_map.keys())
    if not ks:
        raise SystemExit(f"[ERR] no epochXXX_perclass.json found under {args.root} for k{args.kmin}~k{args.kmax}")

    # choose best epoch per k
    best_epoch_per_k = {}
    best_stats_per_k = {}
    perclass_best_per_k = {}  # k -> {names, accs, supp}

    for k in ks:
        best_ep = None
        best_score = None
        best_pack = None

        for ep, path in sorted(perclass_map[k].items()):
            obj = load_json(path)
            names, accs, supp = parse_perclass(obj)
            score = macro_mean(accs) if args.metric == "macro" else weighted_mean(accs, supp)

            if (best_score is None) or (score > best_score):
                best_score = score
                best_ep = ep
                best_pack = (names, accs, supp, path)

        best_epoch_per_k[k] = best_ep
        names, accs, supp, path = best_pack
        perclass_best_per_k[k] = {"names": names, "acc": accs, "support": supp, "path": path}
        best_stats_per_k[k] = {
            "best_epoch": best_ep,
            "score": float(best_score),
            "macro": float(macro_mean(accs)),
            "weighted": float(weighted_mean(accs, supp)),
            "file": path
        }

    # align classes by name (use k=lowest as ref)
    ref_k = ks[0]
    ref_names = perclass_best_per_k[ref_k]["names"]
    ref_supp  = perclass_best_per_k[ref_k]["support"]
    ref_idx = {n:i for i,n in enumerate(ref_names)}
    supp_by_name = {ref_names[i]: int(ref_supp[i]) for i in range(len(ref_names))}

    acc_table = defaultdict(dict)  # class -> {k: acc}
    for k in ks:
        names = perclass_best_per_k[k]["names"]
        accs  = perclass_best_per_k[k]["acc"]
        name_to_acc = {names[i]: float(accs[i]) for i in range(len(names))}
        for n in ref_names:
            if n in name_to_acc:
                acc_table[n][k] = name_to_acc[n]

    # best k per class
    best_per_class = {}
    for n in ref_names:
        avail = sorted(acc_table[n].keys())
        if not avail:
            continue
        best_k = max(avail, key=lambda kk: (acc_table[n][kk], -kk))  # acc 高优先，同分取小k
        best_per_class[n] = {
            "best_k": int(best_k),
            "best_acc": float(acc_table[n][best_k]),
            "support": int(supp_by_name.get(n, 0)),
            "acc_by_k": {str(kk): acc_table[n].get(kk, None) for kk in range(args.kmin, args.kmax+1)}
        }

    out = {
        "k_range": [args.kmin, args.kmax],
        "epochs_considered": sorted(list(ep_set)),
        "best_epoch_per_k": {str(k): int(best_epoch_per_k[k]) for k in ks},
        "best_stats_per_k": {str(k): best_stats_per_k[k] for k in ks},
        "best_per_class": best_per_class,
    }

    out_json_path = os.path.join(args.root, args.out_json)
    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # TSV table: class, support, acc@k2..k6, best_k, best_acc
    out_tsv_path = os.path.join(args.root, args.out_tsv)
    with open(out_tsv_path, "w") as f:
        header = ["class", "support"] + [f"k{kk}" for kk in range(args.kmin, args.kmax+1)] + ["best_k", "best_acc"]
        f.write("\t".join(header) + "\n")
        for n in ref_names:
            info = best_per_class.get(n, None)
            if info is None:
                continue
            row = [n, str(info["support"])]
            for kk in range(args.kmin, args.kmax+1):
                v = info["acc_by_k"].get(str(kk), None)
                row.append("" if v is None else f"{v:.6f}")
            row += [str(info["best_k"]), f"{info['best_acc']:.6f}"]
            f.write("\t".join(row) + "\n")

    # print quick summary
    print("== Best epoch per k ==")
    for k in ks:
        st = best_stats_per_k[k]
        print(f"k={k}: best_epoch={st['best_epoch']:03d} {args.metric}={st['score']:.4f} (macro={st['macro']:.4f}, weighted={st['weighted']:.4f})")
    print(f"\n[OK] wrote:\n  {out_json_path}\n  {out_tsv_path}")

    # show worst classes (by best_acc)
    items = sorted(best_per_class.items(), key=lambda kv: kv[1]["best_acc"])
    print("\n== Worst 10 classes after choosing best k per class ==")
    for n, info in items[:10]:
        print(f"{n:18s} best_k={info['best_k']} best_acc={info['best_acc']:.4f} support={info['support']}")

if __name__ == "__main__":
    main()
