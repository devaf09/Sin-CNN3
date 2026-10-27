"""図: 実験24。手がかりの数と物当ての正解率（3 seed の平均）。条件ごとの線。点線 = 偶然 20%。
実行: python tools/fig_exp24.py   → docs/images/exp24_infer_clues.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp24.txt", encoding="utf-8").read().splitlines()[-1]); rows = r["rows"]
conds = [c for c in rows[0] if c not in ("seed", "k")]; ks = sorted({x["k"] for x in rows})
fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
for i, c in enumerate(conds):
    ax.plot(ks, [100 * np.mean([x[c] for x in rows if x["k"] == k]) for k in ks], "o-" if i else "s--", c="k" if i == 0 else f"C{i - 1}", label=c)
ax.axhline(20, c="gray", ls=":", label="偶然 20%")
ax.set(xlabel="手がかりの数（観点の数）", ylabel="物当ての正解率 [%]（3 seed、各 2000 問）", title="手がかりの数と物当て（掛け算の前の明るさの粗さ）", xticks=ks)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("docs/images/exp24_infer_clues.png")
