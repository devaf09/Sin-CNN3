"""図: 実験20。曲がる処理ごとの「鋭さの増え方」（出口の勝者の割合 − 入口の勝者の割合、点）。点 = 各 seed、棒 = 平均。
実行: python tools/fig_exp20.py   → docs/images/exp20_winner.png"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp20.txt", encoding="utf-8").read().splitlines()[-1])
rows = [x for x in r["rows"] if not any(math.isnan(v) for v in x["gain_each"])]
fig, ax = plt.subplots(figsize=(9, 4.4), dpi=130)
for i, x in enumerate(rows):
    ax.bar(i, 100 * x["gain"][0], 0.6, color="C0" if "飽和" in x["kind"] else ("C1" if "カー" in x["kind"] else "gray"))
    ax.plot([i] * 3, [100 * v for v in x["gain_each"]], "k.", ms=8)
ax.axhline(0, c="k", lw=1, ls="--", label="線形（表計算）の上限 0 点")
ax.set_xticks(range(len(rows)), [x["kind"] for x in rows], fontsize=8)
ax.set(ylabel="鋭さの増え方 [点]（出口の勝者の割合 − 入口）", title="強い道だけを残せるか（カー効果 γ = 300 は学習が発散して除外）"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("docs/images/exp20_winner.png")
