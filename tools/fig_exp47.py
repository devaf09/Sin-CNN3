"""図: 実験47（取り出しの非線形とループ）。左 = 重ねた 1 枚の板（N = 256）、中 = 鍵ごとに別の記憶（N = 16）の p ごとの容量、右 = ループで問いがきれいになるか（N = 256、別々）。
実行: python tools/fig_exp47.py → docs/images/exp47_readout_nonlinear.png"""
import collections, json, math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
R = collections.defaultdict(list)
for l in open("results/exp47.txt", encoding="utf-8"):
    r = json.loads(l); R[r["N"]].append(r)


def curve(N, key):
    Ms = sorted({r["M"] for r in R[N] if key in r})
    v = [np.mean([r[key] for r in R[N] if r["M"] == M]) for M in Ms]
    return Ms, v


fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), dpi=130)
ax = axs[0]
for k, lab, c in [("1段", "1 段（鍵 → 値を 1 枚に重ねる）", "k"), ("2段・重ね p=1", "2 段・重ね p = 1", "C0"), ("2段・重ね p=2", "2 段・重ね p = 2", "C1"),
                  ("2段・重ね p=inf", "2 段・重ね p = ∞（最も明るい点だけ）", "C3")]:
    ax.plot(*curve(256, k), "o-", label=lab, c=c)
ax.set_xscale("log", base=2); ax.set_xlabel("覚えた組の数 M"); ax.set_ylabel("正解率"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
ax.set_title("重ねた 1 枚の板（N = 256、値 16）\n非線形を足しても 1 段を越えない（雑音は重ねた時点で混ざっている）", fontsize=9)
ax = axs[1]
for p, c in zip([1, 2, 3, 4], ["C0", "C1", "C2", "C3"]):
    ax.plot(*curve(16, f"2段・別々 p={p}"), "o-", c=c, label=f"p = {p}")
for p, c in zip([1, 2], ["C0", "C1"]):
    m = 16 * 16 ** (2 * p) / (math.factorial(2 * p) - math.factorial(p) ** 2) / (2 * math.log(16))
    ax.axvline(m, c=c, ls=":", lw=1)
ax.set_xscale("log", base=2); ax.set_xlabel("覚えた組の数 M（記憶の大きさ M × N）"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
ax.set_title("鍵ごとに別の記憶（N = 16、値 16）\np を 1 上げると容量が約 N² 倍（点線 = 式の予想の桁）", fontsize=9)
ax = axs[2]
for p, c in zip([1, 2, "inf"], ["C0", "C1", "C3"]):
    Ms, v = curve(256, f"重なり・別々 p={p} T=5")
    ax.plot(Ms, v, "o-", c=c, label=f"5 回まわした後 p = {p}")
Ms, v = curve(256, "重なり・別々 p=1 T=1"); ax.plot(Ms, v, "k--", label="雑音の入った問い（最初）")
ax.set_xscale("log", base=2); ax.set_xlabel("覚えた組の数 M"); ax.set_ylabel("正しい鍵との重なり |<k, q>|²"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
ax.set_title("ループ（別々の記憶、N = 256）\np ≥ 2 なら 0.25 の問いが 1.0 に戻る。p = 1 は混ざって崩れる", fontsize=9)
fig.tight_layout(); fig.savefig("docs/images/exp47_readout_nonlinear.png")
