"""図: 実験11。覚えさせた組の数 P と、覚えた割合（訓練の正解率）。実線 = 学習率 0.01、点線 = 学習率 0.05（失敗した最初の実行）。
実行: python tools/fig_exp11.py   → docs/images/exp11_grid_learn.png（results/exp11.txt の全部の行から）"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
runs = [json.loads(l) for l in open("results/exp11.txt", encoding="utf-8")]
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=130)
for run in runs:
    lr = run["args"].get("lr", 0.05)
    for i, kind in enumerate(("持ち主の案（縦 sin × 横 cos）", "縦横とも sin と cos", "表（100 × 100 の行列）")):
        rows = [r for r in run["rows"] if r["kind"] == kind]
        if not rows:
            continue
        ax.errorbar([r["P"] for r in rows], [100 * r["acc"][0] for r in rows], yerr=[100 * r["acc"][1] for r in rows],
                    fmt="o-" if lr == 0.01 else "x:", c=f"C{i}", label=f"{kind}、学習する数 {rows[0]['params']}（学習率 {lr}）")
ax.axhline(10, c="gray", ls="--", label="偶然（10 クラス）")
ax.set(xscale="log", xlabel="覚えさせた組の数 P（でたらめな入力 → でたらめなクラス）", ylabel="覚えた割合 [%]（3 seed の平均）",
       title="周波数格子の学習器（曲がる処理なし）は何組覚えられるか", ylim=(0, 105))
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig("docs/images/exp11_grid_learn.png")
