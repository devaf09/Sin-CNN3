"""図: 実験18。関係の数 P と、組がぴったり当たった割合・出口の光のうち答えに入った割合（3 seed の平均 ± 標準誤差）。点線 = 当てはめ 81%·√(8/P)。
実行: python tools/fig_exp18.py   → docs/images/exp18_relation_capacity.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp18.txt", encoding="utf-8").read().splitlines()[-1])
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), dpi=130)
for i, kind in enumerate(("格子 F=16", "格子 F=32", "表")):
    rows = [x for x in r["rows"] if x["kind"] == kind]; P = [x["P"] for x in rows]
    lab = {"格子 F=16": "光の格子（学習する数 1024）", "格子 F=32": "光の格子（4096）", "表": "表（100 × 100 の行列、20000）"}[kind]
    ax[0].errorbar(P, [100 * x["ok"][0] for x in rows], yerr=[100 * x["ok"][1] for x in rows], fmt="o-", label=lab)
    ax[1].errorbar(P, [100 * x["frac"][0] for x in rows], yerr=[100 * x["frac"][1] for x in rows], fmt="o-", label=lab)
Pf = np.array([8, 16, 32, 64, 96]); ax[1].plot(Pf, 80.9 * np.sqrt(8 / Pf), "k:", label="当てはめ 81%·√(8/P)（式ではない）")
for k in range(2):
    ax[k].set(xscale="log", xlabel="入れた関係の数 P（記号 24 × 文脈 4 から）", ylim=(0, 105)); ax[k].set_xticks(Pf, [str(p) for p in Pf]); ax[k].minorticks_off(); ax[k].legend(fontsize=8)
ax[0].set(ylabel="組がぴったり当たった割合 [%]", title="関係の容量: 当たり外れ"); ax[1].set(ylabel="出口の光のうち答えに入った割合 [%]", title="関係の容量: 光の集まり方")
fig.tight_layout(); fig.savefig("docs/images/exp18_relation_capacity.png")
