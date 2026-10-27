"""図: 実験23。存在しない属性を「拡散」で覚えた器と「なし」で覚えた器の、問い合わせの点数（3 seed の平均 ± 標準誤差、灰色 = 表の正解）。
実行: python tools/fig_exp23.py   → docs/images/exp23_absent.png"""
import json, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp23.txt", encoding="utf-8").read().splitlines()[-1]); OBJ = r["OBJ"]; w = np.arange(len(OBJ))
names = list(r["rows"][0]["queries"])
fig, ax = plt.subplots(2, len(names), figsize=(18, 7), dpi=110, sharey=True)
for i, mode in enumerate(("拡散", "なし")):
    rows = [x for x in r["rows"] if x["mode"] == mode]
    for j, nm in enumerate(names):
        S = 100 * np.array([x["queries"][nm]["score"] for x in rows])
        ax[i, j].bar(w - 0.2, S.mean(0), 0.4, yerr=S.std(0, ddof=1) / math.sqrt(len(S)), label="光の器")
        ax[i, j].bar(w + 0.2, 100 * np.array(rows[0]["queries"][nm]["truth"]), 0.4, color="gray", label="表の正解")
        ax[i, j].set_xticks(w, OBJ); ax[i, j].set_title(f"存在しない属性 = {mode} /「{nm}」", fontsize=9); ax[i, j].legend(fontsize=7)
    ax[i, 0].set_ylabel("物の点数 [%]")
fig.tight_layout(); fig.savefig("docs/images/exp23_absent.png")
