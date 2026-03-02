import os
import re
import json
import argparse
from collections import defaultdict

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def safe_get(d, keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def infer_k_from_path(path):
    # ..._k2/... or ...KD_k2/...
    m = re.search(r'[_\-]k(\d+)', path)
    return int(m.group(1)) if m else None

def parse_perclass(perclass_obj):
    """
    兼容不同 perclass.json 格式：
    - dict: { "per_class_acc": [...], "class_names": [...], "support": [...] }
    - dict: { "acc": [...], "name": [...], "count": [...] }
    - list of dict: [ {"id":0,"name":"HO","acc":0.9,"support":123}, ... ]
    """
    if isinstance(perclass_obj, list):
        ids, names, accs, supp = [], [], [], []
        for it in perclass_obj:
            ids.append(int(it.get("id", len(ids))))
            names.append(str(it.get("name", it.get("class_name", ids[-1]))))
            accs.append(float(it.get("acc", it.get("top1", it.get("accuracy", 0.0)))))
            supp.append(int(it.get("support", it.get("count", it.get("n", 0)))))
        return ids, names, accs, supp

    if isinstance(perclass_obj, dict):
        # try common keys
        accs = safe_get(perclass_obj, ["per_class_acc"]) or safe_get(perclass_obj, ["acc"]) or safe_get(perclass_obj, ["top1"]) 
        names = safe_get(perclass_obj, ["class_names"]) or safe_get(perclass_obj, ["name"]) or safe_get(perclass_obj, ["classes"])
        supp = safe_get(perclass_obj, ["support"]) or safe_get(perclass_obj, ["count"]) or safe_get(perclass_obj, ["n"])
        if accs is None:
            # maybe dict keyed by class name
            # e.g., {"HO":{"acc":0.1,"support":10}, ...}
            if all(isinstance(v, dict) for v in perclass_obj.values()):
                names = list(perclass_obj.keys())
                accs = [float(perclass_obj[n].get("acc", perclass_obj[n].get("top1", 0.0))) for n in names]
                supp = [int(perclass_obj[n].get("support", perclass_obj[n].get("count", 0))) for n in names]
            else:
                raise ValueError("Unknown perclass.json format (dict).")
        if names is None:
            names = [str(i) for i in range(len(accs))]
        if supp is None:
            supp = [0 for _ in range(len(accs))]
        ids = list(range(len(accs)))
        return ids, list(map(str, names)), list(map(float, accs)), list(map(int, supp))

    raise ValueError("Unknown perclass.json format.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="work_dir/recognition/tag 的父目录（或任意能递归找到 epochXXX_perclass.json 的目录）")
    ap.add_argument("--epoch", default="000", help="epoch编号，默认000（对应 epoch000_perclass.json）")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=6)
    ap.add_argument("--pattern", default="epoch{epoch}_perclass.json", help="文件名模板，默认 epoch{epoch}_perclass.json")
    ap.add_argument("--out", default="k_sweep_summary.json", help="输出汇总json路径（相对--root）")
    args = ap.parse_args()

    target_name = args.pattern.format(epoch=args.epoch)
    hits = []
    for r, ds, fs in os.walk(args.root):
        if target_name in fs:
            p = os.path.join(r, target_name)
            k = infer_k_from_path(p)
            if k is None:
                continue
            if args.kmin <= k <= args.kmax:
                hits.append((k, p))

    hits.sort(key=lambda x: x[0])
    if not hits:
        raise SystemExit(f"[ERR] no {target_name} found under {args.root}")

    # k -> perclass arrays
    by_k = {}
    class_names_ref = None
    support_ref = None

    for k, path in hits:
        obj = load_json(path)
        ids, names, accs, supp = parse_perclass(obj)
        by_k[k] = {
            "path": path,
            "ids": ids,
            "names": names,
            "acc": accs,
            "support": supp
        }
        if class_names_ref is None:
            class_names_ref = names
            support_ref = supp

    # align by name (稳一点)
    name_to_idx_ref = {n:i for i,n in enumerate(class_names_ref)}
    acc_table = defaultdict(dict)  # class_name -> {k: acc}
    supp_table = {class_names_ref[i]: int(support_ref[i]) for i in range(len(class_names_ref))}

    for k, pkg in by_k.items():
        names = pkg["names"]
        accs = pkg["acc"]
        for i, n in enumerate(names):
            if n in name_to_idx_ref:
                acc_table[n][k] = float(accs[i])

    # best k per class
    best_per_class = {}
    for n in class_names_ref:
        ks = sorted(acc_table[n].keys())
        if not ks:
            continue
        best_k = max(ks, key=lambda kk: (acc_table[n][kk], -kk))  # acc高优先；同acc取小k
        best_per_class[n] = {
            "best_k": best_k,
            "best_acc": acc_table[n][best_k],
            "support": supp_table.get(n, 0),
            "all": {str(kk): acc_table[n].get(kk, None) for kk in range(args.kmin, args.kmax+1)}
        }

    # overall summaries
    overall = {}
    for k in sorted(by_k.keys()):
        # macro mean over classes
        vals = []
        weighted_sum = 0.0
        weighted_n = 0
        for n in class_names_ref:
            a = acc_table[n].get(k, None)
            if a is None:
                continue
            vals.append(a)
            w = supp_table.get(n, 0)
            weighted_sum += a * w
            weighted_n += w
        macro = sum(vals)/len(vals) if vals else 0.0
        weighted = (weighted_sum/weighted_n) if weighted_n > 0 else 0.0
        overall[str(k)] = {"macro_mean": macro, "weighted_mean": weighted, "num_classes": len(vals)}

    out = {
        "k_found": [k for k,_ in hits],
        "epoch": args.epoch,
        "overall": overall,
        "best_per_class": best_per_class,
        "files": {str(k): by_k[k]["path"] for k in sorted(by_k.keys())}
    }

    out_path = os.path.join(args.root, args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # print quick view
    print("== Found k files ==")
    for k, p in hits:
        print(f"k={k} -> {p}")
    print("\n== Overall ==")
    for k in sorted(by_k.keys()):
        o = overall[str(k)]
        print(f"k={k}: macro={o['macro_mean']:.4f} weighted={o['weighted_mean']:.4f}")
    print(f"\n[OK] wrote: {out_path}")

    print("\n== Best k per class (top 10 by worst best_acc) ==")
    items = sorted(best_per_class.items(), key=lambda kv: kv[1]["best_acc"])
    for n, info in items[:10]:
        print(f"{n:15s} best_k={info['best_k']} best_acc={info['best_acc']:.4f} support={info['support']}")

if __name__ == "__main__":
    main()
