"""図: 実験22。通した回数と、勝者の明るさの割合（3 seed の平均）。点線 = 式。
実行: python tools/fig_exp22.py   → docs/images/exp22_sharpen.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp22.txt", encoding="utf-8").read().splitlines()[-1])
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=130)
for i, x in enumerate(r["rows"]):
    k = np.arange(len(x["curve"]))
    ax.errorbar(k, 100 * np.array(x["curve"]), yerr=100 * np.array(x["curve_se"]), fmt="o-", c=f"C{i}", label=x["kind"])
s = np.linspace(0.55, 0.9, 2001); c = [s.mean()]                                   # 式（2 乗）: 入力の割合は 0.55〜0.9 に一様
for _ in range(len(r["rows"][0]["curve"]) - 1):
    s = s ** 2 / (s ** 2 + (1 - s) ** 2); c.append(s.mean())
ax.plot(np.arange(len(c)), 100 * np.array(c), "k:", lw=2, label="式（2 乗）: s → s²/(s² + (1 − s)²)")
ax.axhline(50, c="gray", ls="--", lw=0.8)
ax.set(xlabel="通した回数（0 = 入口）", ylabel="勝者の明るさの割合 [%]", title="鋭さを上げる方法の比べ（重なった 2 つの記号から明るい方を残す）", ylim=(45, 102))
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig("docs/images/exp22_sharpen.png")
