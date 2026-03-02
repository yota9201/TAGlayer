#!/usr/bin/env python3
import argparse, pickle, numpy as np
from pathlib import Path

def load_any(path: Path):
    # 先试 numpy
    try:
        arr = np.load(path, allow_pickle=True)
        # .npz -> 取第一个
        if hasattr(arr, "files") and arr.files:
            arr = arr[arr.files[0]]
        # 0维object标量：里面是真对象
        if isinstance(arr, np.ndarray) and arr.dtype==object and arr.shape==():
            try:
                arr = arr.item()
            except Exception:
                pass
        return arr
    except Exception:
        pass
    # 再试 pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def walk(obj, prefix="", depth=0, maxn=5):
    ind = "  " * depth
    if isinstance(obj, np.ndarray):
        print(f"{ind}{prefix}ndarray shape={obj.shape} dtype={obj.dtype}")
        return
    if isinstance(obj, dict):
        print(f"{ind}{prefix}dict keys({len(obj)}): {list(obj.keys())[:10]}")
        c = 0
        for k,v in obj.items():
            if c>=maxn: 
                print(f"{ind}  ... {len(obj)-maxn} more")
                break
            walk(v, prefix=f"[{k!r}] -> ", depth=depth+1, maxn=maxn)
            c+=1
        return
    if isinstance(obj, (list, tuple)):
        print(f"{ind}{prefix}{type(obj).__name__} len={len(obj)}")
        for i,x in enumerate(obj[:maxn]):
            walk(x, prefix=f"[{i}] -> ", depth=depth+1, maxn=maxn)
        if len(obj)>maxn:
            print(f"{ind}  ... {len(obj)-maxn} more")
        return
    print(f"{ind}{prefix}{type(obj).__name__}: {obj}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    args = ap.parse_args()
    p = Path(args.path)
    obj = load_any(p)
    print(f"[root] type={type(obj).__name__}")
    walk(obj)

if __name__ == "__main__":
    main()
