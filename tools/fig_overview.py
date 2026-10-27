"""図: Sin-CNN3 の器のしくみ（概念図）。入口の波（記号の周波数 + 観点のずらし）→ 位相の板と回折を何列も → 出口を周波数に分けて読む。逆向きにも通せる。
実行: python tools/fig_overview.py   → docs/images/overview_device.png"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
fig, ax = plt.subplots(figsize=(16, 6), dpi=130); ax.set_xlim(0, 16.5); ax.set_ylim(-0.9, 5.2); ax.axis("off")
y = np.linspace(0.6, 4.4, 300)
ax.plot(0.9 + 0.35 * np.cos(2 * math.pi * 6 * (y - 0.6) / 3.8), y, c="C0", lw=1.5)
ax.text(0.9, 4.75, "入口の光\n記号 = 周波数の波\n× 観点のずらし（f1）\n× モード（f2）", ha="center", va="bottom", fontsize=9)
rng = np.random.default_rng(0)
for i, x in enumerate(np.linspace(2.4, 8.4, 7)):
    ax.imshow(rng.uniform(-1, 1, (40, 1)), extent=(x - 0.12, x + 0.12, 0.6, 4.4), cmap="twilight", aspect="auto", vmin=-1, vmax=1)
    if i < 6:
        ax.annotate("", (x + 0.85, 2.5), (x + 0.2, 2.5), arrowprops=dict(arrowstyle="->", color="gray"))
ax.text(5.4, 4.75, "位相の板（学習する）と回折を何列も\n（小さいときは周波数の格子、大きいときは画素ごとの板）", ha="center", va="bottom", fontsize=9)
ax.text(5.4, -0.75, "途中に「明るさを測って出し直す」段を入れると、組み合わせ（語順・両方そろったときだけ）が作れる", ha="center", fontsize=8.5, color="C3")
ax.annotate("", (9.6, 2.5), (8.6, 2.5), arrowprops=dict(arrowstyle="->", lw=2))
ax.text(9.1, 2.75, "レンズ\n（FFT）", ha="center", fontsize=8)
vals = rng.dirichlet(np.ones(8) * 0.3)
ax.barh(np.linspace(0.8, 4.2, 8), vals * 3, height=0.3, left=10.0, color="C2")
for yy, t in zip(np.linspace(0.8, 4.2, 8), ["赤", "緑", "丸い", "甘い", "酸っぱい", "しゃきしゃき", "むしゃむしゃ", "…"]):
    ax.text(9.9, yy, t, ha="right", va="center", fontsize=8)
ax.text(11.4, 4.75, "出口: 値（属性・次の記号）の明るさ\n= 割合・確信度", ha="center", va="bottom", fontsize=9)
ax.add_patch(FancyArrowPatch((11.4, 0.45), (0.9, 0.45), connectionstyle="arc3,rad=-0.08", arrowstyle="->", color="C1", lw=1.5, mutation_scale=15))
ax.text(11.6, 0.2, "← 逆向きに通す = 逆引き（値 → それを持つ物。順向きと同じ明るさ）", ha="left", va="center", fontsize=8.5, color="C1")
fig.tight_layout(); fig.savefig("docs/images/overview_device.png")
