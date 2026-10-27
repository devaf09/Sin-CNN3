"""図: 最近の結果（実験38・40・41・44・45）を 1 枚に。数字は NOTES.md の各節の表から（中央の値、範囲は棒の線）。
実行: python tools/fig_overview_latest.py   → docs/images/overview_latest.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.4), dpi=130)

ax = axs[0]                                                                              # 実験38: 次の文字の当たり
names = ["1-gram", "2-gram", "3-gram", "光 1 段", "光 2 段"]; v = [19.8, 25.9, 33.9, 23.0, 27.4]
ax.bar(names, v, color=["0.7", "0.7", "0.7", "C0", "C3"])
ax.axhline(25.9, ls="--", c="gray", lw=0.8)
for i, x in enumerate(v):
    ax.text(i, x + 0.5, f"{x}", ha="center", fontsize=8)
ax.set_title("実験38: 次の文字の当たり（%）\n明るさを測って出し直す 2 段で 2-gram を越えた", fontsize=9); ax.set_ylim(0, 38)

ax = axs[1]                                                                              # 実験40: 柔らかいの漏れ
tools = ["圧力鍋\n（80% にしたい）", "プレス機\n（0% に）", "鍋\n（0% に）"]
one = [(39 + 47) / 2, (22 + 33) / 2, (23 + 29) / 2]; two = [(42 + 49) / 2, 4.5, 13]
x = range(3)
ax.bar([i - 0.2 for i in x], one, 0.4, label="1 段", color="C0"); ax.bar([i + 0.2 for i in x], two, 0.4, label="2 段", color="C3")
ax.set_xticks(list(x)); ax.set_xticklabels(tools, fontsize=8); ax.legend(fontsize=8)
ax.set_title("実験40: 「柔らかい」と答えた割合（%）\n2 段で、片方の道具だけのときの漏れが減る", fontsize=9)

ax = axs[2]                                                                              # 実験41: モードの数と光
M = [2, 4, 8]; light = [20.5, 13, 6]
ax.plot(M, light, "o-", c="C2")
for m, l in zip(M, light):
    ax.text(m, l + 0.8, f"{l:g}%", ha="center", fontsize=8)
ax.set_xticks(M); ax.set_xlabel("f2 のモードの数（ふつう・最短・考え中・睡眠 …）", fontsize=8); ax.set_ylim(0, 25)
ax.set_title("実験41: 1 つの器で答え方を切り替える\n正確さは保つが、答えに届く光が減る", fontsize=9)

ax = axs[3]                                                                              # 実験45: 初期の板と速さ
plates = ["平ら", "雑音 0.1", "雑音 1", "雑音 ±π", "縞 左高(直線)", "縞 左高(反比例)", "レンズ"]
lo = [250, 150, 75, 50, 100, 100, 100]; hi = [250, 150, 100, 100, 100, 100, 150]
mid = [(a + b) / 2 for a, b in zip(lo, hi)]
ax.barh(plates, mid, xerr=[[m - a for m, a in zip(mid, lo)], [b - m for m, b in zip(mid, hi)]], color=["0.6"] + ["C1"] * 6)
ax.invert_yaxis(); ax.set_xlabel("表との差 0.15 を切った学習回数（少ないほど速い）", fontsize=8); ax.tick_params(labelsize=8)
ax.set_title("実験45: 初期の板の型と学習の速さ\nばらばらな板ほど速い（平らが最も遅い）", fontsize=9)

fig.tight_layout(); fig.savefig("docs/images/overview_latest.png")
