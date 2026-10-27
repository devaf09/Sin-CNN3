"""図: 実験09。上 = 格子の値 G（位相の遅れ）。下 = 曲がる処理（なし・飽和・カー効果）ごとに、入力 a・b・a + b の光の 2 次元マップ（明るさ）と、
右端の出力: 出力(a + b) と 出力(a) + 出力(b) の重ね（なしなら重なる = 1 枚の表と同じ）。
実行: python tools/fig_exp09.py   → docs/images/exp09_lightmap.png（results/exp09.txt の最後の行から）"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp09.txt", encoding="utf-8").read().splitlines()[-1])
rows = r["rows"]
fig = plt.figure(figsize=(16, 4 + 3.4 * len(rows)), dpi=110)
gs = fig.add_gridspec(1 + len(rows), 4)
ax = fig.add_subplot(gs[0, :2])
im = ax.imshow(np.array(r["G"]), cmap="RdBu_r", aspect="auto")
ax.set(title=f"格子の値 G（位相の遅れ [ラジアン]、縦 sin × 横 cos の周波数 1〜{r['args']['F']}、強さはでたらめ）", xlabel="列 x（光は左から右へ）", ylabel="縦の位置 y")
fig.colorbar(im, ax=ax)
for k, row in enumerate(rows):
    for j, (key, t) in enumerate((("I_a", "入力 a（30 番に光の点）"), ("I_b", "入力 b（60〜69 番に弱い帯）"), ("I_ab", "入力 a + b"))):
        ax = fig.add_subplot(gs[1 + k, j])
        ax.imshow(np.sqrt(np.array(row[key])), cmap="magma", aspect="auto")       # 明るさの平方根で表示（暗い所も見えるように）
        ax.set(title=f"曲がる処理: {row['bend']} / {t}", xlabel="列 x", ylabel="y")
    ax = fig.add_subplot(gs[1 + k, 3])
    ax.plot(row["out_ab"], "k-", label="出力(a + b)")
    ax.plot(row["out_sum"], "C1:", lw=2, label="出力(a) + 出力(b)")
    ax.set(title=f"右端の出力（重ね合わせの差 {row['superpos_err']:.1e}）", xlabel="縦の位置 y", ylabel="明るさ")
    ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("docs/images/exp09_lightmap.png")
