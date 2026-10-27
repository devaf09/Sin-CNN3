"""図: 実験12。「どちらか片方だけ」の試験の正解率。列の間 1 画素（最初の試み）と 64 画素（器を直した後）。
点線 = 偶然 50%、破線 = 曲がる処理なしで明るさを読むときの上限 83.3%（干渉を使う。学び 12）。
実行: python tools/fig_exp12.py   → docs/images/exp12_xor.png（results/exp12.txt の全部の行から）"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
runs = [json.loads(l) for l in open("results/exp12.txt", encoding="utf-8")]
kinds = [r["kind"] for r in runs[0]["rows"]]
fig, ax = plt.subplots(figsize=(10, 4.6), dpi=130)
for j, run in enumerate(runs):
    dz = run["args"].get("dz", 1.0)
    acc = {r["kind"]: r["test"] for r in run["rows"]}
    x = np.arange(len(kinds)) + (j - 0.5) * 0.38
    ax.bar(x, [100 * acc[k][0] for k in kinds], 0.36, yerr=[100 * acc[k][1] for k in kinds], label=f"列の間 {dz:g} 画素")
ax.axhline(50, c="gray", ls=":", label="偶然 50%")
ax.axhline(100 * 5 / 6, c="k", ls="--", lw=1, label="曲がる処理なしの上限 83.3%（干渉を使う）")
ax.set_xticks(range(len(kinds)), kinds, fontsize=8)
ax.set(ylabel="試験の正解率 [%]（3 seed の平均 ± 標準誤差）", title="「どちらか片方だけ」（表では解けない課題）", ylim=(0, 105))
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig("docs/images/exp12_xor.png")
