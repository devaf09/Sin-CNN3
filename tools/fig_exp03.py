"""図: 実験03。割り切れる周波数（f = 2）と半端な周波数（f = 2.5）の FFT の大きさ、窓なしとハン窓。式の予想（窓なし）は白抜きの点。
実行: python tools/fig_exp03.py   → docs/images/exp03_leak.png（results/exp03.txt の最後の行から）"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp03.txt", encoding="utf-8").read().splitlines()[-1])
N = r["args"]["N"]; n = np.arange(N); t = np.linspace(0, N, 400)


def geom(d):
    """Σ_{n=0}^{N-1} e^(2πi·d·n/N) を等比数列の和の式で。"""
    q = np.exp(2j * math.pi * d / N); one = np.abs(1 - q) < 1e-12
    return np.where(one, N, (1 - q ** N) / np.where(one, 1, 1 - q))


fig, ax = plt.subplots(2, 2, figsize=(12, 6.4), dpi=130)
for j, f in enumerate((2.0, 2.5)):
    w = 0.5 - 0.5 * np.cos(2 * math.pi * t / N)
    ax[0, j].plot(t, np.cos(2 * math.pi * f * t / N), ":", c="gray", label="cos（連続）")
    ax[0, j].stem(n, np.cos(2 * math.pi * f * n / N), basefmt=" ", label="標本（16 点）")
    ax[0, j].plot(t, w * np.cos(2 * math.pi * f * t / N), c="C2", alpha=0.6, label="ハン窓を掛けたもの")
    ax[0, j].plot(t, w, "--", c="C2", lw=0.8, label="ハン窓")
    ax[0, j].set(title=f"f = {f}（16 点で {f} 周）" + ("" if f == int(f) else " — 端がつながらない"), xlabel="位置 n", ylabel="値")
    ax[0, j].legend(fontsize=7, loc="lower left")
    rows = {x["window"]: x for x in r["rows"] if x["f"] == f}
    ax[1, j].bar(n - 0.2, rows["なし"]["mag"], 0.4, label=f"窓なし（遠くへの漏れ {100 * rows['なし']['leak']:.2f}%）")
    ax[1, j].bar(n + 0.2, rows["ハン窓"]["mag"], 0.4, color="C2", label=f"ハン窓（{100 * rows['ハン窓']['leak']:.3f}%）")
    ax[1, j].plot(n - 0.2, np.abs(0.5 * geom(f - n) + 0.5 * geom(-f - n)), "o", mfc="none", c="k", ls="none", label="式の予想（窓なし）")
    ax[1, j].set(title="FFT の大きさ |X[k]|", xlabel="周波数の番号 k", ylabel="大きさ")
    ax[1, j].legend(fontsize=7)
fig.tight_layout(); fig.savefig("docs/images/exp03_leak.png")
