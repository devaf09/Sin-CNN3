"""図: 実験01（長さ 8 の cos）。左 = 入れた波形、中 = FFT の大きさ、右 = FFT の位相。式の予想は点線・白抜き、計算は実線・塗り。
実行: python tools/fig_exp01.py   → docs/images/exp01_cos8.png（results/exp01.txt の最後の行から作る）"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp01.txt", encoding="utf-8").read().splitlines()[-1])
N, k0 = r["args"]["N"], r["args"]["k0"]
x = np.array(r["x"]); n = np.arange(N); k = np.arange(N)
Xn = np.array([complex(*z) for z in r["X_naive"]])
Xp = np.array([complex(*z) for z in r["X_numpy"]])
Xt = np.array([complex(*z) for z in r["X_torch"]])
Xe = np.zeros(N); Xe[k0] = Xe[N - k0] = N / 2

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), dpi=130)
t = np.linspace(0, N, 400)
ax[0].plot(t, np.cos(2 * math.pi * k0 * t / N), ":", c="gray", label="もとの cos（連続）")
ax[0].stem(n, x, basefmt=" ", label="標本 x[n]（8 点）")
ax[0].set(title=f"入れた波形 x[n] = cos(2π·{k0}·n/{N})", xlabel="位置 n（点の番号）", ylabel="値")
ax[0].legend(fontsize=8, loc="lower left")
w = 0.25
ax[1].bar(k - w, np.abs(Xn), w, label="素朴な和（自作）")
ax[1].bar(k, np.abs(Xp), w, label="numpy.fft.fft")
ax[1].bar(k + w, np.abs(Xt), w, label="torch.fft.fft")
ax[1].plot(k, Xe, "o", mfc="none", c="k", ls="none", ms=8, label="式の予想（N/2 = 4）")
ax[1].set(title="FFT の大きさ |X[k]|", xlabel="周波数の番号 k（8 点で何周するか。4 は ±4、5〜7 は負の周波数 −3〜−1）", ylabel="大きさ")
ax[1].legend(fontsize=8)
big = np.abs(Xn) > 1e-9                                   # 大きさがほぼ 0 の所の位相は意味がない（丸めの誤差の向き）ので描かない
ax[2].plot(k[big], np.angle(Xn[big]), "o", ms=9, label="素朴な和")
ax[2].plot(k[big], np.angle(Xp[big]), "x", ms=9, label="numpy")
ax[2].plot(k[big], np.angle(Xt[big]), "+", ms=12, label="torch")
ax[2].axhline(0, ls=":", c="k", label="式の予想（cos なので 0）")
ax[2].set(title="FFT の位相（大きさがある所だけ）", xlabel="周波数の番号 k", ylabel="位相 [ラジアン]", xlim=(-0.5, N - 0.5), ylim=(-math.pi, math.pi))
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig("docs/images/exp01_cos8.png")
