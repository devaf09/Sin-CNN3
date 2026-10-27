"""図: 実験34。融合の形ごとの、目標（2 つの平均）との差（3 seed の点と平均）と、値に届いた光。
実行: python tools/fig_exp34.py   → docs/images/exp34_fusion.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp34.txt", encoding="utf-8").read().splitlines()[-1]); rows = r["rows"]; ks = list(rows[0]["forms"]); w = np.arange(len(ks))
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), dpi=120)
for j, (m, lab) in enumerate((("dist", "目標（2 つの平均）との差（小さいほど良い）"), ("light", "入れた光のうち値に届いた割合"))):
    V = np.array([[x["forms"][k][m] for k in ks] for x in rows])
    ax[j].bar(w, V.mean(0), 0.6, color=["C2" if V.mean(0)[i] < 0.2 else "C3" for i in range(len(ks))] if m == "dist" else "C0")
    for s in V:
        ax[j].plot(w, s, "k.", ms=5)
    ax[j].set_xticks(w, ks, fontsize=7, rotation=25, ha="right"); ax[j].set(ylabel=lab)
fig.suptitle("リンゴとトマトの格子の融合の形（緑 = うまくいった）"); fig.tight_layout(); fig.savefig("docs/images/exp34_fusion.png")
