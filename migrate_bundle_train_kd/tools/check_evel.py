import os, re, json
import numpy as np

# ===== 你只需要改这里 =====
WORKDIR = "./work_dir/recognition/tag/STGCN_C5_teamMatchup_noSem/eval_arrays"  # 改成你图里那些 epochXXX_*.npy 所在目录
EPOCH = 13 # None=自动用最新；或写成 6 这种数字指定

def find_latest_epoch(workdir):
    pat = re.compile(r"epoch(\d+)_logits\.npy$")
    epochs = []
    for fn in os.listdir(workdir):
        m = pat.search(fn)
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        raise FileNotFoundError("No epochXXX_logits.npy found in workdir")
    return max(epochs)

def load_epoch(workdir, epoch):
    def p(name): return os.path.join(workdir, f"epoch{epoch:03d}_{name}")
    logits = np.load(p("logits.npy")).astype(np.float32)
    labels = np.load(p("labels.npy")).astype(np.int64)
    conf_path = p("confusion.npy")
    confusion = np.load(conf_path) if os.path.exists(conf_path) else None
    perclass_path = p("perclass.json")
    perclass = None
    if os.path.exists(perclass_path):
        with open(perclass_path, "r", encoding="utf-8") as f:
            perclass = json.load(f)
    return logits, labels, confusion, perclass

def softmax(logits):
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)

def main():
    epoch = find_latest_epoch(WORKDIR) if EPOCH is None else int(EPOCH)
    logits, labels, confusion, perclass = load_epoch(WORKDIR, epoch)

    N, K = logits.shape
    p = softmax(logits)
    pred = p.argmax(axis=1)
    correct = (pred == labels)

    p_true = p[np.arange(N), labels]
    ce = -np.log(np.clip(p_true, 1e-12, 1.0))

    print("===== BASIC =====")
    print("epoch =", epoch)
    print("N =", N, "num_class =", K)
    print("Top1 =", float(correct.mean()))
    print("CE(mean) =", float(ce.mean()))
    print("p_true(mean) ~ exp(-CE) check =", float(p_true.mean()))

    print("\n===== TAIL CONTRIBUTION (does a small set dominate CE?) =====")
    idx = np.argsort(-ce)  # high->low
    total = ce.sum()
    for frac in [0.01, 0.05, 0.10]:
        k = max(1, int(N * frac))
        share = ce[idx[:k]].sum() / (total + 1e-12)
        print(f"Top {int(frac*100)}% samples contribute {share*100:.1f}% of total CE")

    print("\n===== CONFIDENCE DIAGNOSTICS =====")
    p_max = p.max(axis=1)
    wrong_conf = (~correct) & (p_max > 0.9)
    print("Wrong & confident (pmax>0.9):", float(wrong_conf.mean()), "count", int(wrong_conf.sum()))

    corr_low = correct & (p_true < 0.8)
    print("Correct but p_true<0.8:", float(corr_low.mean()), "count", int(corr_low.sum()))

    # margin = top1 prob - top2 prob (or logit margin)
    p_sorted = np.sort(p, axis=1)
    margin = p_sorted[:, -1] - p_sorted[:, -2]
    print("Prob margin mean =", float(margin.mean()), "std =", float(margin.std()))

    print("\n===== PER-CLASS (computed from logits/labels) =====")
    acc_c = np.zeros(K, dtype=np.float32)
    ce_c = np.zeros(K, dtype=np.float32)
    cnt_c = np.zeros(K, dtype=np.int64)

    for c in range(K):
        m = (labels == c)
        cnt = int(m.sum())
        cnt_c[c] = cnt
        if cnt > 0:
            acc_c[c] = correct[m].mean()
            ce_c[c] = ce[m].mean()

    worst = np.argsort(-ce_c)
    print("Worst classes by CE (class, cnt, acc, ce):")
    for c in worst[:10]:
        if cnt_c[c] == 0: 
            continue
        print(int(c), int(cnt_c[c]), float(acc_c[c]), float(ce_c[c]))

    print("\n===== TOP CONFUSION PAIRS =====")
    # 如果你已经保存了 confusion.npy 就直接用；否则用 pred/labels 现算一个
    if confusion is None:
        confusion = np.zeros((K, K), dtype=np.int64)
        for y, yhat in zip(labels, pred):
            confusion[y, yhat] += 1

    pairs = []
    for i in range(K):
        for j in range(K):
            if i != j and confusion[i, j] > 0:
                pairs.append((int(confusion[i, j]), i, j))
    pairs.sort(reverse=True)
    for ct, i, j in pairs[:12]:
        print(ct, "true", int(i), "pred", int(j))

    print("\n===== TOP HARD SAMPLES (indices) =====")
    # 如果你有 names 文件也可以加进来；目前先输出 index、ce、是否正确、p_true、pmax
    for t in range(min(20, N)):
        i = int(idx[t])
        print(t, "idx", i, "ce", float(ce[i]), "correct", bool(correct[i]),
              "p_true", float(p_true[i]), "pmax", float(p_max[i]), "pred", int(pred[i]), "label", int(labels[i]))

    # 额外：如果你保存的 perclass.json 存在，也打印一下（可和我们算的对照）
    if perclass is not None:
        print("\n===== perclass.json exists (show keys) =====")
        print(list(perclass.keys())[:20])

if __name__ == "__main__":
    main()
