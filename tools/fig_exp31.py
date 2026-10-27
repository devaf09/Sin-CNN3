"""図: 実験31。左 = 単語ごとの格子の大きさと表との差（形 1・形 2、3 seed、点線 = 共有の格子 1 枚の実験21）。右 = 融合の仕方と、2 つの平均への近さ（小さいほど良い）。
実行: python tools/fig_exp31.py   → docs/images/exp31_word_grids.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp31.txt", encoding="utf-8").read().splitlines()[-1]); rows = r["rows"]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), dpi=130)
Fws = sorted({x["Fw"] for x in rows}); w = np.arange(len(Fws))
for j, form in enumerate((1, 2)):
    T = [[x["tv"] for x in rows if x["form"] == form and x["Fw"] == f] for f in Fws]
    ax[0].bar(w + (j - 0.5) * 0.38, [np.mean(t) for t in T], 0.36, label=f"形 {form}")
    for i, t in enumerate(T):
        ax[0].plot([i + (j - 0.5) * 0.38] * len(t), t, "k.", ms=5)
ax[0].axhline(0.13, c="gray", ls=":", label="共有の格子 1 枚（実験21、学習する数 1024）")
ax[0].set_xticks(w, [f"1 単語 {4 * f * f} 個（{16 * f * f} バイト）" for f in Fws]); ax[0].set(ylabel="順引きの分布と表の差（小さいほど良い）", title="単語ごとに格子 1 枚"); ax[0].legend(fontsize=8)
for j, k in enumerate(("位相の平均", "2 枚に分けて足す")):
    v = [x["fuse"][k] for x in rows if x["Fw"] == max(Fws)]
    ax[1].bar(j, np.mean(v), 0.5, color=f"C{j + 2}"); ax[1].plot([j] * len(v), v, "k.", ms=5)
ax[1].set_xticks([0, 1], ["2 枚の位相を平均した 1 枚", "2 枚に光を分けて通し、足す"])
ax[1].set(ylabel="リンゴとトマトの平均との差（小さいほど良い）", title="2 つの単語の格子の融合（1 単語 1024 バイト、形 1・2、3 seed）")
fig.tight_layout(); fig.savefig("docs/images/exp31_word_grids.png")
