"""図: 実験07。(A) ぐるっと回る畳み込みと普通の畳み込み、(B) 相関の山、(C) 前のプロジェクトの記憶の読み出し（回る版とゼロを詰めた版）。
実行: python tools/fig_exp07.py   → docs/images/exp07_circular.png（results/exp07.txt の最後の行から）"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp07.txt", encoding="utf-8").read().splitlines()[-1])
N = r["args"]["N"]
fig, ax = plt.subplots(2, 2, figsize=(12, 9), dpi=120)
ax = ax.ravel()
A = r["A"]
ax[0].bar(np.arange(N) - 0.2, A["circ"], 0.4, label="FFT の掛け算（ぐるっと回る）")
ax[0].bar(np.arange(N + 2) + 0.2, A["linear"], 0.4, label="普通の畳み込み = ゼロを詰めた FFT")
ax[0].annotate("8・9 番のはみ出し（15, 8）が\n0・1 番に回り込む", (0.2, 16), (1.5, 24), fontsize=8, arrowprops=dict(arrowstyle="->"))
ax[0].set(title="(A) x = [1..8] と h = [1, 1, 1] の畳み込み", xlabel="位置", ylabel="値"); ax[0].set_ylim(0, 30); ax[0].legend(fontsize=7, loc="upper right")
ax[1].stem(np.arange(N), r["B"]["corr"], basefmt=" ")
ax[1].set(title="(B) 鍵と、3 つずらした鍵の相関", xlabel="ずらし s", ylabel="相関（山 = 8 = 鍵の長さ）")
for j, (key, t) in enumerate((("I", "(C) 記憶の読み出し: ぐるっと回る（8 × 8）"), ("Ip", "(C) 同じ鍵と点: ゼロを詰めた（16 × 16）"))):
    im = ax[2 + j].imshow(np.log10(np.array(r["C"][key]) + 1), cmap="magma", vmin=0, vmax=3.7)
    ax[2 + j].set(title=t); ax[2 + j].set_xticks([]); ax[2 + j].set_yticks([])
fig.colorbar(im, ax=ax[2:], label="明るさ（log10(1 + 明るさ)）。山 = 4096")
fig.savefig("docs/images/exp07_circular.png", bbox_inches="tight")
