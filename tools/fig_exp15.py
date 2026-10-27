"""図: 実験15（関係性概念）。
exp15_stage1.png: 第 1 段階（δ = 0、seed 0）— 入力の例、格子（学習前・後）、入力 A の光の 2 次元マップ（学習前・後）、4 つの入力の出力の窓（学習前・後）、学習の経過。
exp15_stage2.png: 第 2 段階 — ずらしの量 δ と、組がぴったり当たった割合・答えの窓に入った光・出口に届いた光（seed 0）、式: 入力どうしの重なり。
                  δ = 8 で、同じ A の入力が「ずらしなし」「ずらしあり」で別の点に光る様子（学習後）。
実行: python tools/fig_exp15.py（results/exp15.txt の最後の行と results/exp15_maps/ から）"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp15.txt", encoding="utf-8").read().splitlines()[-1])
SYM = list("ABCDEFGHXYZ"); N = r["args"]["N"]; pos = np.arange(N)
rows = r["rows"]
r0 = [x for x in rows if x["delta"] == 0 and x["seed"] == 0][0]; z = np.load(r0["maps"])

fig = plt.figure(figsize=(16, 11), dpi=110); gs = fig.add_gridspec(3, 4)
ax = fig.add_subplot(gs[0, 0])
for n, (s, d, t) in enumerate(r0["pairs"]):
    ax.plot(pos, z["in_I"][n] + 1.2 * n, label=f"入力 {s}（答え {','.join(t)}）")
ax.set(title="入力の例（光の点、上へずらして表示）", xlabel="縦の位置 y", yticks=[]); ax.legend(fontsize=7)
for j, (k, t) in enumerate((("G_before", "格子（学習前）"), ("G_after", "格子（学習後）"))):
    ax = fig.add_subplot(gs[0, 1 + j]); im = ax.imshow(z[k], cmap="RdBu_r", aspect="auto"); fig.colorbar(im, ax=ax)
    ax.set(title=f"{t}: 位相の遅れ [ラジアン]", xlabel="列 x", ylabel="y")
ax = fig.add_subplot(gs[0, 3])
for x in rows:
    c = np.array(x["curve"]); ax.plot(c[:, 0], c[:, 1], c=plt.cm.viridis(min(x["delta"], 8) / 8), lw=0.8, alpha=0.8)
ax.set(title="学習の経過（損失。色 = ずらしの量 δ、0 = 紫 〜 8 = 黄）", xlabel="学習の回数", ylabel="損失", yscale="log")
for j, (k, t) in enumerate((("map_before", "学習前"), ("map_after", "学習後"))):
    ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
    ax.imshow(np.sqrt(z[k][0].astype(float)), cmap="magma", aspect="auto")
    ax.set(title=f"入力 A の光の 2 次元マップ（{t}、明るさの平方根）", xlabel="列 x（列の間 64 画素）", ylabel="y")
    ax.set_yticks([8 + 84 * i / 10 for i in range(11)], SYM, fontsize=7)
for n, (s, d, t) in enumerate(r0["pairs"]):
    ax = fig.add_subplot(gs[2, n]); w = np.arange(len(SYM))
    ax.bar(w - 0.2, np.array(r0["win_before"][n]) / max(r0["win_before"][n]), 0.4, color="gray", label="学習前")
    ax.bar(w + 0.2, np.array(r0["win_after"][n]) / max(r0["win_after"][n]), 0.4, color=["C3" if c in t else "C0" for c in SYM], label="学習後（赤 = 答え）")
    ax.set_xticks(w, SYM); ax.set(title=f"入力 {s} → 答え {','.join(t)}：出力の窓の明るさ（最大 = 1）"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("docs/images/exp15_stage1.png")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), dpi=120)
deltas = sorted({x["delta"] for x in rows if x["delta"] > 0})
m = lambda key, i, d: [x[key][i] for x in rows if x["delta"] == d]
ax[0].errorbar(deltas, [100 * np.mean(m("after", 0, d)) for d in deltas], yerr=[100 * np.std(m("after", 0, d), ddof=1) / math.sqrt(3) for d in deltas], fmt="o-", label="組がぴったり当たった割合")
ax[0].errorbar(deltas, [100 * np.mean(m("after", 1, d)) for d in deltas], yerr=[100 * np.std(m("after", 1, d), ddof=1) / math.sqrt(3) for d in deltas], fmt="s-", label="出口の光のうち答えの窓に入った割合")
trans = [float((np.load(f"results/exp15_maps/delta{d:g}_seed0.npz")["out_after"].sum(1) / np.load(f"results/exp15_maps/delta{d:g}_seed0.npz")["in_I"].sum(1)).mean()) for d in deltas]
ax[0].plot(deltas, [100 * t for t in trans], "^:", c="k", label="入れた光のうち出口（100 画素）に届いた割合（seed 0）")
ax[0].set(xscale="log", xlabel="ずらしの量 δ（100 画素で何周）", ylabel="[%]（3 seed の平均）", title="第 2 段階: ずらしの量と成績", ylim=(0, 105)); ax[0].set_xticks(deltas, [f"{d:g}" for d in deltas]); ax[0].minorticks_off(); ax[0].legend(fontsize=7)
s = np.exp(-((pos - 50) / 1.5) ** 2); dd = np.logspace(-0.7, 1.3, 50)
ov = [abs((s ** 2 * np.exp(2j * math.pi * d * pos / N)).sum()) / (s ** 2).sum() for d in dd]
ax[1].plot(dd, ov, "k-"); ax[1].set(xscale="log", xlabel="ずらしの量 δ", ylabel="重なり（1 = 同じ光）", title="式: ずらす前と後の入力の光の重なり\n（光の量を保つ器は、出力の重なりも同じに保つ）")
r8 = [x for x in rows if x["delta"] == 8 and x["seed"] == 0][0]; w = np.arange(len(SYM))
for n, (sym, d, t) in enumerate(r8["pairs"]):
    if sym == "A":
        v = np.array(r8["win_after"][n]); off = -0.2 if d == 0 else 0.2
        ax[2].bar(w + off, v / v.max(), 0.4, label=f"A、ずらし {'なし' if d == 0 else 'あり（δ = 8）'} → 答え {','.join(t)}")
ax[2].set_xticks(w, SYM); ax[2].set(title="δ = 8（学習後）: 同じ A が、ずらしで別の点に光る", ylabel="窓の明るさ（最大 = 1）"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig("docs/images/exp15_stage2.png")
