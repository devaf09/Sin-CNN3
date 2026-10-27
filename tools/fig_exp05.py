"""図: 実験05。行ごとに 1 つの組み合わせ。左 = 入力 x・格子の波 g・掛けたもの x·g の波形、右 = x·g の FFT の大きさ（式の予想は白抜きの点）。
実行: python tools/fig_exp05.py   → docs/images/exp05_mix.png（results/exp05.txt の最後の行から）"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp05.txt", encoding="utf-8").read().splitlines()[-1])
N = r["args"]["N"]; n = np.arange(N); k = np.arange(N)
expect = [{2: 8, 8: 8, 24: 8, 30: 8}, {2: 8, 8: 8, 24: 8, 30: 8}, {2: 8, 10: 8, 22: 8, 30: 8},
          {1: 8, 5: 16, 9: 8, 23: 8, 27: 16, 31: 8}]           # 差と和（と、その負の周波数）に N/4 ずつ
fig, ax = plt.subplots(len(r["rows"]), 2, figsize=(13, 2.6 * len(r["rows"])), dpi=120, gridspec_kw=dict(width_ratios=[1.2, 1]))
for i, row in enumerate(r["rows"]):
    x, g = np.array(row["x"]), np.array(row["g"])
    ax[i, 0].plot(n, x, "o-", ms=3, lw=0.8, label="入力 x")
    ax[i, 0].plot(n, g, "s-", ms=3, lw=0.8, label="格子の波 g")
    ax[i, 0].plot(n, x * g, "k-", lw=1.6, label="掛けたもの x·g")
    ax[i, 0].set(title=row["case"], xlabel="位置 n", ylabel="値"); ax[i, 0].legend(fontsize=7, loc="upper right", ncol=3)
    Y = np.array([complex(*z) for z in row["Y"]])
    e = np.zeros(N)
    for kk, v in expect[i].items():
        e[kk] = v
    ax[i, 1].bar(k, np.abs(Y), 0.6, label="x·g の FFT の大きさ")
    ax[i, 1].plot(k, e, "o", mfc="none", c="k", ls="none", label="式の予想（差と和に N/4 = 8）")
    ax[i, 1].axvline(N / 2, ls=":", c="gray")
    ax[i, 1].set(xlabel="周波数の番号 k（点線より右は負の周波数）", ylabel="大きさ", ylim=(0, 18)); ax[i, 1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("docs/images/exp05_mix.png")
