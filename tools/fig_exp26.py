"""図: 実験26。文字列「リンゴ」の意味（果物 / 人）の点数。光の器（3 seed の平均）と表から計算した点数。
実行: python tools/fig_exp26.py   → docs/images/exp26_homonym.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp26.txt", encoding="utf-8").read().splitlines()[-1]); names = list(r["rows"][0]["queries"]); w = np.arange(len(names))
S = 100 * np.array([[x["queries"][n]["score"][:2] for n in names] for x in r["rows"]]).mean(0); T = 100 * np.array([r["rows"][0]["queries"][n]["truth"][:2] for n in names])
fig, ax = plt.subplots(figsize=(12, 4.6), dpi=130)
ax.bar(w - 0.3, S[:, 0], 0.2, color="C3", label="果物のリンゴ（光の器）"); ax.bar(w - 0.1, T[:, 0], 0.2, color="C3", alpha=0.4, label="果物（表）")
ax.bar(w + 0.1, S[:, 1], 0.2, color="C0", label="人のリンゴ（光の器）"); ax.bar(w + 0.3, T[:, 1], 0.2, color="C0", alpha=0.4, label="人（表）")
ax.set_xticks(w, names, fontsize=8); ax.set(ylabel="意味の点数 [%]", title="2 つの意味を持つ言葉「リンゴ」: 重ね合わせ + 手がかりの掛け算"); ax.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig("docs/images/exp26_homonym.png")
