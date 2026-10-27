"""図: 実験27。左 = 文ごとの「たどった明るさ」（文字列 → 物 → 仲間 → 手がかり、対数、3 seed の点と平均）書き込みの前と後。
右 = 書き込み後の「リンゴ」の意味の点数（3 seed の平均）。
実行: python tools/fig_exp27.py   → docs/images/exp27_unknown_sense.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp27.txt", encoding="utf-8").read().splitlines()[-1]); rows = r["rows"]; OBJ = r["OBJ"]
names = list(rows[0]["before"]); w = np.arange(len(names))
fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=120, gridspec_kw=dict(width_ratios=[1.6, 1]))
for k, (key, c, off) in enumerate((("before", "C0", -0.2), ("after", "C1", 0.2))):
    F = np.array([[x[key][n]["fit"] for n in names] for x in rows])
    ax[0].bar(w + off, np.exp(np.log(F).mean(0)), 0.4, color=c, label="書き込みの前" if key == "before" else "書き込みの後（リンゴ → 未知 1 → 人間 を足す）")
    for s in range(len(rows)):
        ax[0].plot(w + off, F[s], "k.", ms=5)
ax[0].set_yscale("log"); ax[0].set_xticks(w, names, fontsize=7, rotation=12)
ax[0].set(ylabel="たどった明るさ（どれだけ合うか、対数）", title="知っている意味に合わない文は、とても暗い（= 驚き）"); ax[0].legend(fontsize=8)
qs = ["リンゴを食べた", "リンゴは楽しそうに手を振った（持ち主の例）", "リンゴが話した"]; v = np.arange(len(OBJ))
for i, q in enumerate(qs):
    S = 100 * np.array([x["after"][q]["sense"] for x in rows]).mean(0)
    ax[1].bar(v + (i - 1) * 0.27, S, 0.27, label=q)
ax[1].set_xticks(v, OBJ, fontsize=8); ax[1].set(ylabel="意味の点数 [%]", title="書き込みの後の「リンゴ」の意味"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("docs/images/exp27_unknown_sense.png")
