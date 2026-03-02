import os
import re
import json
import argparse
from typing import Dict, Any, Tuple, List


# -----------------------------
# Helpers
# -----------------------------

def parse_epochs(spec: str) -> List[int]:
    """
    spec examples:
      "0-39"
      "0,1,2,10-20"
    """
    spec = spec.strip()
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a <= b:
                out.extend(list(range(a, b + 1)))
            else:
                out.extend(list(range(a, b - 1, -1)))
        else:
            out.append(int(part))
    # unique preserve order
    seen = set()
    out2 = []
    for e in out:
        if e not in seen:
            seen.add(e)
            out2.append(e)
    return out2


def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)


def find_exp_dir(root: str, pattern: str, k: int) -> str:
    """
    pattern default:
      "STGCN_C4_processed_semTA_TAG_KD_k{K}_ball"
    """
    name = pattern.replace("{K}", str(k))
    p = os.path.join(root, name)
    if not os.path.isdir(p):
        raise FileNotFoundError(f"Experiment dir not found for k={k}: {p}")
    return p


def load_perclass_json(p: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """
    Normalize per-class metrics.

    Accepted common formats:
      A) list of dict rows:
         [{"class":0,"support":249,"acc":0.95,"name":"HO"}, ...]
      B) dict mapping class_id(str/int) -> dict:
         {"0":{"acc":...,"support":...,"name":...}, ...}
      C) dict with key "per_class" / "class_acc" etc, nested as A/B

    Output:
      rows[cid] = {"acc": float, "support": int|None, "name": str|None}
    """
    with open(p, "r") as f:
        obj = json.load(f)

    def norm_from_list(lst):
        rows = {}
        for r in lst:
            if not isinstance(r, dict):
                continue
            cid = r.get("class_id", r.get("class", r.get("id")))
            if cid is None:
                continue
            cid = int(cid)
            acc = r.get("acc", r.get("accuracy", r.get("top1", r.get("recall"))))
            sup = r.get("support", r.get("count", r.get("n")))
            name = r.get("name", r.get("class_name"))
            if acc is None:
                continue
            rows[cid] = {
                "acc": float(acc),
                "support": int(sup) if sup is not None else None,
                "name": name
            }
        return rows

    def norm_from_dict(dct):
        rows = {}
        for k, v in dct.items():
            try:
                cid = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                acc = v.get("acc", v.get("accuracy", v.get("top1", v.get("recall"))))
                sup = v.get("support", v.get("count", v.get("n")))
                name = v.get("name", v.get("class_name"))
            else:
                acc = v
                sup = None
                name = None
            if acc is None:
                continue
            rows[cid] = {
                "acc": float(acc),
                "support": int(sup) if sup is not None else None,
                "name": name
            }
        return rows

    # A)
    if isinstance(obj, list):
        rows = norm_from_list(obj)
        return rows, obj

    # C) nested key
    if isinstance(obj, dict):
        for key in ["per_class", "class_acc", "perclass", "per_class_acc"]:
            if key in obj and isinstance(obj[key], (list, dict)):
                nested = obj[key]
                if isinstance(nested, list):
                    rows = norm_from_list(nested)
                else:
                    rows = norm_from_dict(nested)
                return rows, obj

        # B) flat dict
        rows = norm_from_dict(obj)
        if rows:
            return rows, obj

    raise ValueError(f"Unrecognized perclass json format: {p}")


def compute_score(rows: Dict[int, Dict[str, Any]], metric: str) -> float:
    """
    metric:
      - "macro": average over classes
      - "weighted": weighted by support (if support missing -> fall back macro)
    """
    if not rows:
        return float("-inf")

    accs = []
    weights = []
    for cid, v in rows.items():
        a = float(v["acc"])
        sup = v.get("support", None)
        accs.append(a)
        weights.append(int(sup) if sup is not None else None)

    if metric == "macro":
        return sum(accs) / max(1, len(accs))

    if metric == "weighted":
        if all(w is not None for w in weights):
            total = sum(weights)
            if total <= 0:
                return sum(accs) / max(1, len(accs))
            return sum(a * w for a, w in zip(accs, weights)) / total
        # support 缺失则退化为 macro
        return sum(accs) / max(1, len(accs))

    raise ValueError(f"Unknown metric: {metric}")


def epoch_file(exp_dir: str, epoch: int) -> str:
    return os.path.join(exp_dir, "eval_arrays", f"epoch{epoch:03d}_perclass.json")


# -----------------------------
# Main analysis
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g. work_dir/recognition/tag")
    ap.add_argument("--kmin", type=int, default=2)
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--epochs", required=True, help="e.g. 0-39 or 0,1,2,10-20")
    ap.add_argument("--metric", choices=["weighted", "macro"], default="weighted")
    ap.add_argument("--pattern", default="STGCN_C4_processed_semTA_TAG_KD_k{K}_ball",
                    help="experiment dir name pattern under root, use {K} placeholder")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_tsv", required=True)
    args = ap.parse_args()

    epochs = parse_epochs(args.epochs)
    ks = list(range(args.kmin, args.kmax + 1))

    # 1) For each k, find best epoch by metric
    per_k = {}
    class_name = {}
    class_support = {}

    for k in ks:
        exp_dir = find_exp_dir(args.root, args.pattern, k)

        best = None  # (score, epoch, rows_path, rows)
        scores_by_epoch = {}

        for ep in epochs:
            p = epoch_file(exp_dir, ep)
            if not os.path.isfile(p):
                continue
            rows, _raw = load_perclass_json(p)
            sc = compute_score(rows, args.metric)
            scores_by_epoch[ep] = sc

            if best is None or sc > best[0]:
                best = (sc, ep, p, rows)

        if best is None:
            raise FileNotFoundError(
                f"No perclass json found for k={k} in epochs={epochs}. "
                f"Check: {os.path.join(exp_dir,'eval_arrays')}"
            )

        best_score, best_epoch, best_path, best_rows = best

        # collect class name/support
        for cid, v in best_rows.items():
            if v.get("name") and cid not in class_name:
                class_name[cid] = v["name"]
            if v.get("support") is not None:
                class_support[cid] = int(v["support"])

        per_k[k] = {
            "exp_dir": exp_dir,
            "best_epoch": best_epoch,
            "best_score": float(best_score),
            "best_file": best_path,
            "scores_by_epoch": {str(ep): float(sc) for ep, sc in sorted(scores_by_epoch.items())},
            "per_class": {str(cid): {"acc": float(v["acc"]),
                                    "support": v.get("support", None),
                                    "name": v.get("name", None)} for cid, v in best_rows.items()}
        }

    # union classes from all best_rows
    all_cids = sorted({int(cid) for k in per_k for cid in per_k[k]["per_class"].keys()})

    # 2) Build per-class comparison table (using each k's best epoch per above)
    per_class_table = []
    for cid in all_cids:
        row = {
            "class_id": cid,
            "class_name": class_name.get(cid, ""),
            "support": class_support.get(cid, None)
        }
        # collect accs
        accs = []
        for k in ks:
            v = per_k[k]["per_class"].get(str(cid), None)
            a = float(v["acc"]) if v is not None else None
            row[f"k{k}"] = a
            if a is not None:
                accs.append((k, a))
        if accs:
            maxa = max(a for _, a in accs)
            bestks = [k for k, a in accs if abs(a - maxa) < 1e-12]
            row["best_k"] = "/".join(map(str, bestks))
            row["best_acc"] = float(maxa)
        else:
            row["best_k"] = ""
            row["best_acc"] = None
        per_class_table.append(row)

    # 3) Dump outputs
    out = {
        "root": args.root,
        "pattern": args.pattern,
        "k_range": [args.kmin, args.kmax],
        "epochs": epochs,
        "metric": args.metric,
        "per_k_best": per_k,
        "per_class_table": per_class_table
    }

    ensure_dir(args.out_json)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    ensure_dir(args.out_tsv)
    header = ["class_id", "class_name", "support"] + [f"k{k}" for k in ks] + ["best_k", "best_acc"]
    with open(args.out_tsv, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in per_class_table:
            vals = [
                str(r["class_id"]),
                str(r["class_name"]),
                "" if r["support"] is None else str(r["support"]),
            ]
            for k in ks:
                a = r.get(f"k{k}", None)
                vals.append("" if a is None else f"{a:.6f}")
            vals.append(str(r.get("best_k", "")))
            ba = r.get("best_acc", None)
            vals.append("" if ba is None else f"{ba:.6f}")
            f.write("\t".join(vals) + "\n")

    print(f"[OK] wrote: {args.out_json}")
    print(f"[OK] wrote: {args.out_tsv}")
    print("=== Best epoch per k ===")
    for k in ks:
        print(f"  k={k} best_epoch={per_k[k]['best_epoch']:03d} best_{args.metric}={per_k[k]['best_score']:.6f}")


if __name__ == "__main__":
    main()
