"""図: 実験17。左 = 回ごとの「答えの記号に入った光」（入れた光 = 1、3 seed）と式 p^回（p = 1 回目の値）。右 = 枝分かれ（seed 0）の回ごとの記号の明るさ。
実行: python tools/fig_exp17.py   → docs/images/exp17_chain.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp17.txt", encoding="utf-8").read().splitlines()[-1]); SYM = list("ABCDEFGHXYZ")
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), dpi=130, gridspec_kw=dict(width_ratios=[1, 1.4]))
for i, name in enumerate(("一本道", "枝分かれ")):
    for x in [x for x in r["rows"] if x["chain"] == name]:
        h = [y["hop"] for y in x["hops"] if y["light_on_expect"] is not None]; v = [y["light_on_expect"] for y in x["hops"] if y["light_on_expect"] is not None]
        ax[0].plot(h, v, "o-", c=f"C{i}", alpha=0.7, label=name if x["seed"] == 0 else None)
        ax[0].plot(h, [v[1] ** k for k in h], ":", c=f"C{i}", lw=1)
ax[0].set(xlabel="何回目（出口の光をそのまま次の入力に）", ylabel="答えの記号に入った光（入れた光 = 1）", title="関係をつなげる: 回ごとの光（点線 = 式 p^回）", ylim=(0, 1.05)); ax[0].legend()
x = [x for x in r["rows"] if x["chain"] == "枝分かれ" and x["seed"] == 0][0]; w = np.arange(len(SYM))
for k, y in enumerate(x["hops"][:4]):
    ax[1].bar(w + (k - 1.5) * 0.2, y["sym_light"], 0.2, label=f"{y['hop']} 回目（答え {','.join(y['expect'])}）")
ax[1].set_xticks(w, SYM); ax[1].set(title="枝分かれ A → B, C / B → D / C → E / D → F / E → G（seed 0）", ylabel="記号の周波数の明るさ"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig("docs/images/exp17_chain.png")
