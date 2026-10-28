"""図: 学び 59（語板を数万語に）の説明用。
  exp59_what_is_goban.png : 語板とは何か（1 語 = 1 枚の光の器。入口 = 観点、出口 = 値の明るさ）と、数えた表との比べ
  exp59_results.png       : 名詞 2 万・動詞 3000 の当たり（語板 / 平らにした表 / 語を見ない）
  exp60_mix.png           : 材料の混ぜ方（会話の重み）と、会話・全部の確かめ用での当たり
実行: python tools/fig_exp59.py"""
import json, math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]

# 1. 語板とは何か
fig, ax = plt.subplots(figsize=(15, 6.2), dpi=130); ax.set_xlim(0, 15); ax.set_ylim(0, 6.2); ax.axis("off")
views = ["どんな", "〜を", "〜が", "〜で", "〜に"]
for i, v in enumerate(views):
    y = 5.0 - i * 0.9
    ax.text(0.2, y, f"観点「{v}」の光", va="center", fontsize=10)
    xs = np.linspace(1.9, 2.9, 200); ax.plot(xs, y + 0.18 * np.sin(2 * math.pi * (i + 1) * (xs - 1.9) * 2), c=f"C{i}", lw=1.3)
    ax.annotate("", (3.6, 3.2), (3.0, y), arrowprops=dict(arrowstyle="->", color=f"C{i}", lw=0.8))
ax.add_patch(FancyBboxPatch((3.7, 1.0), 2.6, 4.4, boxstyle="round,pad=0.05", fc="#eef3ff", ec="C0"))
rng = np.random.default_rng(1)
for k, x in enumerate(np.linspace(4.0, 6.0, 6)):
    ax.imshow(rng.uniform(-1, 1, (30, 1)), extent=(x - 0.07, x + 0.07, 1.3, 5.1), cmap="twilight", aspect="auto", vmin=-1, vmax=1, zorder=3)
ax.set_xlim(0, 15); ax.set_ylim(0, 6.2)
ax.text(5.0, 5.6, "語板「ラーメン」\n（位相の板と回折。学習で板の模様が決まる）", ha="center", fontsize=10)
ax.annotate("", (7.1, 3.2), (6.4, 3.2), arrowprops=dict(arrowstyle="->", lw=2)); ax.text(6.75, 3.45, "レンズ", ha="center", fontsize=8)
bars = [("食べる", .55), ("作る", .2), ("頼む", .1), ("調べる", .05), ("その他", .1)]
for j, (w, p) in enumerate(bars):
    y = 4.6 - j * 0.6; ax.barh(y, p * 4, left=8.2, height=0.42, color="C2"); ax.text(8.1, y, w, ha="right", va="center", fontsize=9)
    ax.text(8.25 + p * 4, y, f"{int(p * 100)}%", va="center", fontsize=8)
ax.text(9.2, 5.3, "出口: 観点「〜を」の光を入れたとき\n「ラーメン を ○○」の ○○ の明るさ", ha="center", fontsize=9.5)
ax.add_patch(FancyBboxPatch((10.9, 1.0), 3.9, 4.4, boxstyle="round,pad=0.05", fc="#f6f6f6", ec="0.5"))
ax.text(12.85, 5.0, "比べ: 数えた表（デジタルの辞書）", ha="center", fontsize=10)
tbl = "ラーメン を 食べる … 552 回\nラーメン を 作る  … 201 回\nラーメン を 頼む  … 98 回\n…\n→ 割合にすると同じ形"
ax.text(11.1, 3.0, tbl, va="center", fontsize=9.5, zorder=4)
ax.text(7.5, 0.35, "語板は、この「数えた表」と同じ答えを光で出すように学ぶ。違いは、同じ器のまま 逆引き（値 → 語）・掛け算・融合・たどる ができること。\n（数字はたとえ。本物の数は非公開の材料から数えている）", ha="center", fontsize=9, color="0.25")
fig.tight_layout(); fig.savefig("docs/images/exp59_what_is_goban.png"); plt.close(fig)

# 2. 実験59 の結果
rows = [json.loads(l) for l in open("results/exp59.txt", encoding="utf-8")]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), dpi=130)
for ax, kind in zip(axs, ["名詞", "動詞"]):
    groups = [("全部の材料の\n確かめ用", "全部"), ("会話の\n確かめ用", "会話")]
    for k, (lab, key, c) in enumerate([("語を見ない", "語を見ない", "0.6"), ("語板（光の器）", "語板", "C3"), ("平らにした表（デジタル）", "平らにした表", "C0")]):
        vals = []
        for _, g in groups:
            name = {"語を見ない": f"語を見ない {g}", "語板": f"語板 {'全部の確かめ用' if g == '全部' else '会話の確かめ用'}", "平らにした表": f"平らにした表 {g}"}[key]
            vals.append(np.mean([r[kind][name]["bits"] for r in rows]))
        ax.bar(np.arange(2) + (k - 1) * 0.27, vals, 0.27, color=c, label=lab)
        for i, v in enumerate(vals):
            ax.text(i + (k - 1) * 0.27, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(2)); ax.set_xticklabels([g for g, _ in groups]); ax.set_ylabel("交差エントロピー（ビット、低いほど良い）"); ax.legend(fontsize=8)
    ax.set_title(f"実験59: {kind}（{'2 万語' if kind == '名詞' else '3000 語'}）。語板は表との差 {'0.03〜0.05' if kind == '名詞' else '0.04〜0.05'} ビット", fontsize=9.5)
fig.tight_layout(); fig.savefig("docs/images/exp59_results.png"); plt.close(fig)

# 3. 実験60 の混ぜ方（Web 小説を含む最新の行）
r60 = [json.loads(l) for l in open("results/exp60.txt", encoding="utf-8")][-1]["res"]
names = [n for n in ["会話だけ k=5", "全部（重み 1） k=20", "会話 × 3 + ほか k=20", "会話 × 10 + ほか k=20", "会話 × 30 + ほか k=20", "会話 × 100 + ほか k=20"] if n in r60]
fig, ax = plt.subplots(figsize=(11, 4.4), dpi=130)
x = np.arange(len(names))
ax.plot(x, [r60[n]["会話"] for n in names], "o-", c="C1", label="会話の確かめ用")
ax.plot(x, [r60[n]["全部"] for n in names], "s-", c="C0", label="全部の材料の確かめ用")
for i, n in enumerate(names):
    ax.text(i, r60[n]["会話"] + 0.05, f"{r60[n]['会話']:.2f}", ha="center", fontsize=8, color="C1"); ax.text(i, r60[n]["全部"] - 0.12, f"{r60[n]['全部']:.2f}", ha="center", fontsize=8, color="C0")
ax.set_xticks(x); ax.set_xticklabels([n.split(" k=")[0].replace(" + ほか", "\n+ ほか") for n in names], fontsize=8)
ax.set_ylabel("平らにした表の交差エントロピー（ビット）"); ax.legend(fontsize=8); ax.set_ylim(1.85, 3.8)
ax.set_title("実験60: 材料の混ぜ方。会話の数に重みを掛けると、会話の当たりが上がり、全部の当たりはほとんど下がらない\n（会話だけだと、会話以外の文で大きく崩れる）", fontsize=9.5)
fig.tight_layout(); fig.savefig("docs/images/exp60_mix.png"); plt.close(fig)
print("ok")
