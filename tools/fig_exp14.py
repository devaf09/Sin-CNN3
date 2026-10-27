"""図: 実験14。色を分けた「どちらか片方だけ」の試験の正解率（各 seed の点と平均）。点線 = 偶然 50%、破線 = 曲がる処理なしの上限 75%。
実行: python tools/fig_exp14.py   → docs/images/exp14_xor_colors.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp14.txt", encoding="utf-8").read().splitlines()[-1]); rows = r["rows"]
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=130)
for i, x in enumerate(rows):
    ax.bar(i, 100 * x["test"][0], 0.6, color="C1" if "カー" in x["kind"] else ("C0" if "飽和" in x["kind"] else "gray"))
    ax.plot([i] * len(x["test_each"]), [100 * v for v in x["test_each"]], "k.", ms=8)
ax.axhline(50, c="gray", ls=":", label="偶然 50%"); ax.axhline(75, c="k", ls="--", lw=1, label="曲がる処理なしの上限 75%（色が違うので干渉は使えない）")
ax.set_xticks(range(len(rows)), [x["kind"] for x in rows], fontsize=8)
ax.set(ylabel="試験の正解率 [%]（棒 = 3 seed の平均、点 = 各 seed）", title="色を分けた「どちらか片方だけ」: 曲がる処理（色をまたいで効く）の効果", ylim=(0, 105)); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("docs/images/exp14_xor_colors.png")
