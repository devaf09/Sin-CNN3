"""図: 実験21。問い合わせごとの、物の点数（観点ごとの逆引きの明るさの掛け算、3 seed の平均 ± 標準誤差）と、表から計算した正解の点数。
実行: python tools/fig_exp21.py   → docs/images/exp21_viewpoint.png"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp21.txt", encoding="utf-8").read().splitlines()[-1]); OBJ = r["OBJ"]
names = list(r["rows"][0]["queries"]); w = np.arange(len(OBJ))
fig, ax = plt.subplots(1, len(names), figsize=(18, 4.4), dpi=120, sharey=True)
for j, nm in enumerate(names):
    S = np.array([x["queries"][nm]["score"] for x in r["rows"]]) * 100
    ax[j].bar(w - 0.2, S.mean(0), 0.4, yerr=S.std(0, ddof=1) / math.sqrt(len(S)), label="光の器（逆引きの掛け算）")
    ax[j].bar(w + 0.2, 100 * np.array(r["rows"][0]["queries"][nm]["truth"]), 0.4, color="gray", label="表から計算した正解")
    ax[j].set_xticks(w, OBJ, fontsize=8); ax[j].set_title(f"「{nm}」", fontsize=8); ax[j].legend(fontsize=7)
ax[0].set_ylabel("物の点数 [%]（合計 100）")
fig.suptitle("観点周波数の逆引き: 手がかりから「どの物の可能性が高いか」")
fig.tight_layout(); fig.savefig("docs/images/exp21_viewpoint.png")
