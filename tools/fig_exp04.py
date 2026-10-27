"""図: 実験04。exp04_board.png = 4 × 4 の格子マップ（cos の板・sin の板）。
exp04_shadow.png = 入力ごとに、入力の縞・cos の板の光影・sin の板の光影・両方を合わせた明るさ √(cos² + sin²)。
実行: python tools/fig_exp04.py（results/exp04.txt の最後の行から）"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp04.txt", encoding="utf-8").read().splitlines()[-1])
N, F = r["args"]["N"], r["F"]
y, x = np.mgrid[0:N, 0:N]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), dpi=130)
for k, (name, fn) in enumerate((("cos", np.cos), ("sin", np.sin))):
    board = np.block([[fn(2 * math.pi * (fx * x + fy * y) / N) for fx in F] for fy in F])
    ax[k].imshow(board, cmap="gray", vmin=-1, vmax=1)
    ax[k].set_xticks([N * j + N / 2 - 0.5 for j in range(4)], [f"横 {f}" for f in F])
    ax[k].set_yticks([N * i + N / 2 - 0.5 for i in range(4)], [f"縦 {f}" for f in F])
    for p in range(1, 4):
        ax[k].axhline(N * p - 0.5, c="red", lw=0.8); ax[k].axvline(N * p - 0.5, c="red", lw=0.8)
    ax[k].set_title(f"{name} の板（区画 16 × 16 画素、周波数 = 16 画素で何周するか）")
fig.tight_layout(); fig.savefig("docs/images/exp04_board.png")

rows = r["rows"]
fig, ax = plt.subplots(len(rows), 4, figsize=(13, 3.0 * len(rows)), dpi=120)
for i, row in enumerate(rows):
    u = np.cos(2 * math.pi * (row["a"] * x + row["b"] * y) / N + math.radians(row["phase_deg"]))
    ax[i, 0].imshow(u, cmap="gray", vmin=-1, vmax=1); ax[i, 0].set_title(row["input"], fontsize=9)
    ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
    C, S = np.array(row["cos"]), np.array(row["sin"])
    for j, (M, t, cm, lo) in enumerate(((C, "cos の板の光影", "RdBu_r", -128), (S, "sin の板の光影", "RdBu_r", -128),
                                       (np.hypot(C, S), "両方を合わせた明るさ √(cos²+sin²)", "magma", 0))):
        ax[i, j + 1].imshow(M, cmap=cm, vmin=lo, vmax=128)
        for (p, q), v in np.ndenumerate(M):
            ax[i, j + 1].text(q, p, f"{v:.0f}", ha="center", va="center", fontsize=8, color="k" if j < 2 else ("k" if v > 70 else "w"))
        ax[i, j + 1].set_xticks(range(4), [f"横{f}" for f in F]); ax[i, j + 1].set_yticks(range(4), [f"縦{f}" for f in F])
        ax[i, j + 1].set_title(t, fontsize=9)
fig.suptitle("入力の縞を 4 × 4 の格子に当てた光影（赤 = 明るい、青 = 影。式の予想と差 4e-14 以下）")
fig.tight_layout(); fig.savefig("docs/images/exp04_shadow.png")
