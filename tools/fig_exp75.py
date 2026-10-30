"""実験75 の図: 器の大きさごとの学習曲線（確かめ用の当たり・語板 6 枚の損失）と計算時間。学習回数の見積もりと計算量の把握に（持ち主の依頼 2026-10-30）。
出力: docs/figs/exp75_learning_curves.png、docs/learning_curves_75.md（見積もりの表）。実行: python tools/fig_exp75.py"""
import json, math, sys
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
R = [json.loads(l) for l in open("results/exp75.txt", encoding="utf-8")]
R.sort(key=lambda d: (d["vals"], d["rows"] * d["plates"], d["rows"]))
lab = lambda d: f"縦{d['rows']}×板{d['plates']}" + (f"・値{d['vals']}" if d["vals"] != 45 else "")
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
# (1) 確かめ用の当たり（ビット）と回数
for k, d in enumerate(R):
    xs = [c["steps"] for c in d["curve"]]; ys = [c["bits"] - d["table"] for c in d["curve"]]
    ax[0].plot(xs, ys, "o-", label=lab(d), color=f"C{k}")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_ylim(0.015, 2)
ax[0].set_xlabel("学習の回数"); ax[0].set_ylabel("表との差（ビット、確かめ用の材料）"); ax[0].set_title("① 回数と当たり（下ほど表に近い）"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")
# (2) 語板 6 枚の学習の損失（いちばん大きい器と、いまの器）
for d, ls in ((R[0], "--"), (max((x for x in R if x["vals"] == 45), key=lambda x: x["rows"] * x["plates"]), "-")):
    for k, (w, lg) in enumerate(d["loss_boards"].items()):
        ax[1].plot(range(1, len(lg) + 1), lg, ls, color=f"C{k}", lw=1.2, label=f"{w}（{lab(d)}）" if ls == "-" else None)
ax[1].set_xscale("log"); ax[1].set_xlabel("学習の回数"); ax[1].set_ylabel("学習の損失（自然対数）")
ax[1].set_title("② 語板 6 枚の学習曲線（実線 = 大きい器、点線 = いまの器）\n左ほどよく出る語、右ほどまれな語"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3, which="both")
# (3) 計算時間（語板 1 枚・1 回あたり）
names = [lab(d) for d in R]; us = [d["timing"]["us_per_board_step"] for d in R]
ax[2].bar(range(len(R)), us, color=[f"C{k}" for k in range(len(R))])
for k, u in enumerate(us):
    ax[2].text(k, u, f"{u:.0f} µs", ha="center", va="bottom", fontsize=9)
ax[2].set_xticks(range(len(R))); ax[2].set_xticklabels(names, rotation=20, fontsize=8)
ax[2].set_ylabel("語板 1 枚・学習 1 回あたりの時間（µs、RTX 3060）"); ax[2].set_title("③ 計算時間（縦 × 板 にほぼ比例）")
plt.tight_layout(); plt.savefig("docs/figs/exp75_learning_curves.png", dpi=110)
# 見積もりの表: 最後の当たりから 0.02 ビット以内に入る回数、3 万語・10 万語の時間
L = ["# 学習曲線と計算量（実験75）", "", "名詞 6000 語（観点 9）、学習率 0.03、途中で止めずに 2000 回。時間は RTX 3060 1 枚の実測。図: [figs/exp75_learning_curves.png](figs/exp75_learning_curves.png)", "",
     "| 器 | 表との差（2000 回） | 0.02 ビット以内に入る回数 | 0.05 以内 | 語板 1 枚・1 回 | 3 万語を 500 回 | 10 万語を 500 回 |", "|---|---|---|---|---|---|---|"]
for d in R:
    fin = d["curve"][-1]["bits"]
    reach = lambda e: next((c["steps"] for c in d["curve"] if c["bits"] <= fin + e), "−")
    u = d["timing"]["us_per_board_step"]
    L.append(f"| {lab(d)} | {fin - d['table']:.3f} | {reach(0.02)} | {reach(0.05)} | {u:.0f} µs | {u * 3e4 * 500 / 60e6:.0f} 分 | {u * 1e5 * 500 / 60e6:.0f} 分 |")
L += ["", "- 時間は学習だけ（表づくり・材料の読み込みは別）。GPU 2 枚なら半分。",
      "- 損失の曲線は、どの器もはじめの 100 回で大きく下がり、500 回でほぼ平ら。大きい器（実線）は、どの語板も 10 回目あたりから、いまの器（点線）より低い所を通る。"]
open("docs/learning_curves_75.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
