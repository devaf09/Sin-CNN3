"""図: 実験33。縦の幅 N と置き方ごとの、順引きの表との差（3 seed の点と平均）。
実行: python tools/fig_exp33.py   → docs/images/exp33_renumber.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp33.txt", encoding="utf-8").read().splitlines()[-1]); rows = r["rows"]
Ns = sorted({x["N"] for x in rows}, reverse=True); modes = ["そのまま", "詰める", "詰めて並べ直す"]; w = np.arange(len(Ns))
fig, ax = plt.subplots(figsize=(9, 4.4), dpi=130)
for j, m in enumerate(modes):
    T = [[x["tv"] for x in rows if x["N"] == n and x["mode"] == m] for n in Ns]
    ax.bar(w + (j - 1) * 0.27, [np.mean(t) for t in T], 0.25, label=m)
    for i, t in enumerate(T):
        ax.plot([i + (j - 1) * 0.27] * len(t), t, "k.", ms=4)
ov = {n: next(x["overlap"] for x in rows if x["N"] == n and x["mode"] == "そのまま") for n in Ns}
ax.set_xticks(w, [f"N = {n}\n（そのままの重なり {100 * ov[n]:.0f}%）" for n in Ns])
ax.set(ylabel="順引きの分布と表の差（小さいほど良い）", title="周波数の置き方と縦の幅（使う周波数は 37 個）"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("docs/images/exp33_renumber.png")
