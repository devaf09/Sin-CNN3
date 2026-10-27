"""図: 実験02。上段 = 3 つの信号の波形、下段 = FFT（norm="backward"）の実部と虚部。式の予想は白抜きの点。
実行: python tools/fig_exp02.py   → docs/images/exp02_norm_sin.png"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
N = 8; n = np.arange(N); t = np.linspace(0, N, 400)
sig = [("cos(2π·n/8)", lambda s: np.cos(2 * math.pi * s / N), {1: 4, 7: 4}),
       ("sin(2π·n/8)", lambda s: np.sin(2 * math.pi * s / N), {1: -4j, 7: 4j}),
       ("2·cos(2π·2n/8 + π/3)", lambda s: 2 * np.cos(2 * math.pi * 2 * s / N + math.pi / 3),
        {2: 8 * np.exp(1j * math.pi / 3), 6: 8 * np.exp(-1j * math.pi / 3)})]
fig, ax = plt.subplots(2, 3, figsize=(13, 6), dpi=130)
for j, (name, fn, peaks) in enumerate(sig):
    x = fn(n)
    ax[0, j].plot(t, fn(t), ":", c="gray"); ax[0, j].stem(n, x, basefmt=" ")
    ax[0, j].set(title=f"入れた波形 {name}", xlabel="位置 n", ylabel="値")
    X = np.fft.fft(x, n=N, norm="backward")                 # 長さ N、正規化なし（明示）
    e = np.zeros(N, complex)
    for k, v in peaks.items():
        e[k] = v
    ax[1, j].bar(n - 0.18, X.real, 0.36, label="実部（cos の成分）")
    ax[1, j].bar(n + 0.18, X.imag, 0.36, label="虚部（sin の成分）")
    ax[1, j].plot(n - 0.18, e.real, "o", mfc="none", c="k", ls="none", label="式の予想")
    ax[1, j].plot(n + 0.18, e.imag, "o", mfc="none", c="k", ls="none")
    ax[1, j].axhline(0, c="k", lw=0.5)
    ax[1, j].set(title='FFT（norm="backward"）', xlabel="周波数の番号 k（後半は負の周波数）", ylabel="値", ylim=(-8.5, 8.5))
    ax[1, j].legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig("docs/images/exp02_norm_sin.png")
