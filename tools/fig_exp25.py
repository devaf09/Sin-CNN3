"""図: 実験25。青空文庫で数えた割合を覚えた器の問い合わせ（3 seed の平均、上位 5 つ）と、数えた割合から計算した点数。
実行: python tools/fig_exp25.py   → docs/images/exp25_aozora_attr.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
r = json.loads(open("results/exp25.txt", encoding="utf-8").read().splitlines()[-1]); OBJ = r["OBJ"]; names = list(r["rows"][0]["queries"])
fig, ax = plt.subplots(2, 3, figsize=(15, 7), dpi=110); ax = ax.ravel()
for j, nm in enumerate(names):
    S = 100 * np.array([x["queries"][nm]["score"] for x in r["rows"]]).mean(0); T = 100 * np.array(r["rows"][0]["queries"][nm]["truth"])
    idx = np.argsort(T)[::-1][:5]; w = np.arange(5)
    ax[j].bar(w - 0.2, S[idx], 0.4, label="光の器"); ax[j].bar(w + 0.2, T[idx], 0.4, color="gray", label="数えた割合から計算")
    ax[j].set_xticks(w, [OBJ[i] for i in idx]); ax[j].set_title(f"「{nm}」"); ax[j].legend(fontsize=7)
fig.suptitle("青空文庫の「形容詞 + 物の名前」から覚えた観点周波数の器（点数 [%]、上位 5 つ）")
fig.tight_layout(); fig.savefig("docs/images/exp25_aozora_attr.png")
