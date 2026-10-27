"""図: 実験19。学習済みの格子（カー効果 γ = 2）に 4 通りの入力を通した光の 2 次元マップ（3 色の明るさの合計、平方根で表示）。
左 = カー効果あり、中 = 同じ格子でカー効果を切る、右 = 右端の窓の明るさ。
実行: python tools/fig_exp19.py   → docs/images/exp19_kerr_maps.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp19.txt", encoding="utf-8").read().splitlines()[-1]); z = np.load(r["maps"])
fig, ax = plt.subplots(4, 3, figsize=(14, 12), dpi=110, gridspec_kw=dict(width_ratios=[1, 1, 0.6]))
for n, row in enumerate(r["rows"]):
    for j, (k, t) in enumerate((("kerr", "カー効果あり"), ("off", "同じ格子でカー効果を切る"))):
        m = np.sqrt(z[k][n].astype(float).sum(0))
        ax[n, j].imshow(m, cmap="magma", aspect="auto")
        for y0, y1, c in ((20, 40, "cyan"), (60, 80, "lime")):
            ax[n, j].plot([99.5, 99.5], [y0, y1], c=c, lw=4)
        ax[n, j].set(title=f"入力: {row['case']} / {t}", xlabel="列 x（列の間 64 画素）", ylabel="y")
    w = np.arange(2)
    ax[n, 2].bar(w - 0.2, row["win_kerr"], 0.4, label="カー効果あり"); ax[n, 2].bar(w + 0.2, row["win_off"], 0.4, label="切る")
    ans = 1 if row["case"] in ("A だけ", "B だけ") else 0
    ax[n, 2].set_xticks(w, ["窓 0（水色）", "窓 1（緑）"]); ax[n, 2].set(title=f"右端の窓の明るさ（正しい答え: 窓 {ans}）"); ax[n, 2].legend(fontsize=7)
fig.suptitle(f"カー効果が光をどう動かすか（試験の正解率: あり {100 * r['acc_kerr']:.0f}%、同じ格子で切ると {100 * r['acc_off']:.0f}%）")
fig.tight_layout(); fig.savefig("docs/images/exp19_kerr_maps.png")
