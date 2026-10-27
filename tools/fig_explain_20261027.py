"""説明用の図（2026-10-27、持ち主の依頼「詳しく説明して」「図付きで」）。
(1) explain_superposition.png: 曲がる処理のない器では、要素の光を足すと答えも足し算になり、どちらの道具も出さない答えは出ない（小さな例）。
(2) explain_scale.png: 規模を広げたとき（実験28）、板の作り方と初期の位相の大きさ・結果。
実行: python tools/fig_explain_20261027.py"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
RES = ["ジュース", "煮る", "食べる", "柔らかい"]
# (1) 小さな例: 器 = 複素数の表 T（線形）。プレス機の光 a → ジュース、鍋の光 b → 煮る・食べる に送るよう作る。柔らかい には誰も光を送らない。
Ta = np.array([0.95, 0.05, 0.05, 0.0]) * np.exp(1j * np.array([0.3, 1.0, 2.0, 0]))
Tb = np.array([0.05, 0.8, 0.55, 0.0]) * np.exp(1j * np.array([2.5, 0.2, 1.1, 0]))
Ia, Ib = np.abs(Ta) ** 2, np.abs(Tb) ** 2; Iab = np.abs((Ta + Tb) / math.sqrt(2)) ** 2
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), dpi=130, sharey=True)
for j, (I, t) in enumerate(((Ia, "プレス機（圧力）だけ"), (Ib, "鍋（熱）だけ"), (Iab, "圧力鍋（圧力 + 熱の光を足して入れる）"))):
    ax[j].bar(range(4), I, color=["C0", "C1", "C2", "C3"]); ax[j].set_xticks(range(4), RES); ax[j].set(title=t)
ax[0].set_ylabel("出口の明るさ")
ax[2].annotate("どちらも 0 なので\n足しても 0", (3, 0.01), (2.2, 0.5), arrowprops=dict(arrowstyle="->"), fontsize=9)
fig.suptitle("曲がる処理がない器: 出口の場 = プレス機の場 + 鍋の場。どちらも光を送らない答え（柔らかい）は、足しても出ない", fontsize=10)
fig.tight_layout(); fig.savefig("docs/images/explain_superposition.png")
# (2) 規模: 実験28 の結果と、初期の位相の大きさ（格子は 4F² 個の強さを足すので、初期値 0.1 だと位相の大きさ ≈ 0.1 × 2F）
rows = []
for f, lab in (("results/exp28b.txt", None), ("results/exp28c.txt", "初期値 1/F")):
    try:
        for x in json.loads(open(f, encoding="utf-8").read().splitlines()[-1])["rows"]:
            F = x["F"]; ph = 0.1 if F == 0 else (0.1 / F if lab else 0.1) * 2 * F
            rows.append((("画素ごとの板" if F == 0 else f"格子 F={F}") + (f"（{lab}）" if lab else ""), x["q1"], x["q2"], ph, x["params"]))
    except FileNotFoundError:
        pass
fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), dpi=130); w = np.arange(len(rows))
ax[0].bar(w - 0.2, [100 * r[1] for r in rows], 0.4, label="雪は何色？型"); ax[0].bar(w + 0.2, [100 * r[2] for r in rows], 0.4, label="白いものは？型（上位 5 の重なり）")
ax[0].axhline(70.3, c="gray", ls=":", label="一番多い値を答える（雪は何色？の比べ）")
ax[0].set_xticks(w, [f"{r[0]}\n学習する数 {r[4]}" for r in rows], fontsize=7); ax[0].set(ylabel="[%]", title="名詞 80 × 観点 9 を覚えられたか（実験28）"); ax[0].legend(fontsize=7)
ax[1].bar(w, [r[3] for r in rows], 0.5, color="C4"); ax[1].set_yscale("log")
ax[1].set_xticks(w, [r[0] for r in rows], fontsize=7); ax[1].set(ylabel="初期の位相の大きさの目安 [ラジアン]（対数）", title="初期の位相の大きさ（大きいと光が最初からでたらめに散る）")
fig.tight_layout(); fig.savefig("docs/images/explain_scale.png")
