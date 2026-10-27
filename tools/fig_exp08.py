"""図: 実験08。左 = 組の数 M と取り出しの正解率（実線 = 実験 3 seed の平均 ± 標準誤差、点線 = 式の予想）。右 = 雑音の分散 × N（実験と式）。
実行: python tools/fig_exp08.py   → docs/images/exp08_memory_capacity.png（results/exp08.txt の最後の行から）"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp08.txt", encoding="utf-8").read().splitlines()[-1])
N = r["args"]["N"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), dpi=130)
for i, kind in enumerate(("HRR", "FHRR", "前型", "前型+鍵1", "前型+ゼロ")):
    rows = [x for x in r["rows"] if x["kind"] == kind]
    M = [x["M"] for x in rows]; c = f"C{i}"
    ax[0].errorbar(M, [100 * x["acc"][0] for x in rows], yerr=[100 * x["acc"][1] for x in rows], fmt="o-", c=c, ms=4, label=kind)
    ax[0].plot(M, [100 * x["pred"] for x in rows], ":", c=c, lw=2)
    ax[1].plot(M, [N * x["var_wrong"][0] for x in rows], "o-", c=c, ms=4, label=f"{kind}（実験）")
    ax[1].plot(M, [N * x["pred_var"][1] for x in rows], ":", c=c, lw=2)
ax[0].axhline(100, c="k", lw=1, ls="--", label="デジタルの辞書（正確な鍵）")
ax[0].axhline(100 / 256, c="gray", lw=1, ls="--", label="偶然（1/256）")
ax[0].set(xscale="log", xlabel=f"重ね書きした組の数 M（信号の長さ N = {N}）", ylabel="取り出しの正解率 [%]", title="容量の線（実線 = 実験、点線 = 式の予想）")
ax[0].legend(fontsize=7)
ax[1].set(xscale="log", yscale="log", xlabel="重ね書きした組の数 M", ylim=(0.3, 1000), ylabel="ほかの候補の雑音の分散 × N（M = 1 の前型+鍵 1 は 0 で枠の外）", title="雑音は組の数に比例して増える（点線 = 式）")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig("docs/images/exp08_memory_capacity.png")
