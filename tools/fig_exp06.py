"""図: 実験06。横軸 = 重ねた本数 K（対数）、縦軸 = 読み出しの相対誤差（対数）。左 = 雑音 σ = 0.1、右 = 4 ビットに丸める。
実線 = 実験（3 seed の平均、ひげ = 標準誤差）、点線 = 式の予想（2σ/√N ÷ 振幅の平均）。
実行: python tools/fig_exp06.py   → docs/images/exp06_capacity_noise.png（results/exp06.txt の最後の行から）"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp06.txt", encoding="utf-8").read().splitlines()[-1])
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4), dpi=130, sharey=True)
for j, noise in enumerate(("σ=0.1", "4 ビット")):
    for mode, c in (("1 本ごとの振幅", "C0"), ("明るさ一定", "C3")):
        rows = [x for x in r["rows"] if x["mode"] == mode and x["noise"] == noise]
        K = [x["K"] for x in rows]
        rel = [100 * x["rel"][0] for x in rows]
        se = [100 * x["rel"][1] for x in rows]
        pred = [100 * x["pred"][0] * x["rel"][0] / x["err"][0] for x in rows]      # 予想の誤差を同じ振幅の平均で割る
        ax[j].errorbar(K, rel, yerr=se, fmt="o-", c=c, label=f"{mode}（実験）")
        ax[j].plot(K, pred, ":", c=c, lw=2, label=f"{mode}（式の予想）")
    ax[j].set(xscale="log", yscale="log", xlabel="重ねた数の本数 K（N = 64 の信号 1 本に）", title=f"雑音: {noise}")
    ax[j].set_xticks([1, 4, 16, 31], ["1", "4", "16", "31"])
    ax[j].legend(fontsize=8)
ax[0].set_ylabel("読み出しの相対誤差 [%]（誤差 ÷ 振幅の平均）")
fig.tight_layout(); fig.savefig("docs/images/exp06_capacity_noise.png")
