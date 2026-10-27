"""図: 実験16。記号の表し方（光の点 = 実験15 / 整数の周波数の波 = 実験16）と、ずらしの量 δ。
左 = 組がぴったり当たった割合、右 = 入れた光のうち答えに届いた割合（= 出口に届いた割合 × 出口の光のうち答えの割合）。3 seed の平均。
実行: python tools/fig_exp16.py   → docs/images/exp16_relations_freq.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), dpi=130)
for line in open("results/exp16.txt", encoding="utf-8"):
    r = json.loads(line); dz = r["args"]["dz"]; ds = sorted({x["delta"] for x in r["rows"] if x["delta"] > 0})
    ok = [100 * np.mean([x["after"][0] for x in r["rows"] if x["delta"] == d]) for d in ds]
    got = [100 * np.mean([x["after"][1] * x["after"][2] for x in r["rows"] if x["delta"] == d]) for d in ds]
    ax[0].plot(ds, ok, "o-", label=f"周波数の波（実験16、列の間 {dz:g}）"); ax[1].plot(ds, got, "o-", label=f"周波数の波（実験16、列の間 {dz:g}）")
r = json.loads(open("results/exp15.txt", encoding="utf-8").read().splitlines()[-1]); ds = sorted({x["delta"] for x in r["rows"] if x["delta"] > 0})
ax[0].plot(ds, [100 * np.mean([x["after"][0] for x in r["rows"] if x["delta"] == d]) for d in ds], "s:", c="gray", label="光の点（実験15、列の間 64）")
tr = {d: float((np.load(f"results/exp15_maps/delta{d:g}_seed0.npz")["out_after"].sum(1) / np.load(f"results/exp15_maps/delta{d:g}_seed0.npz")["in_I"].sum(1)).mean()) for d in ds}
ax[1].plot(ds, [100 * np.mean([x["after"][1] for x in r["rows"] if x["delta"] == d]) * tr[d] for d in ds], "s:", c="gray", label="光の点（実験15、届いた割合は seed 0）")
for k in range(2):
    ax[k].set(xscale="log", xlabel="ずらしの量 δ（100 画素で何周）", ylim=(0, 105)); ax[k].set_xticks([0.25, 0.5, 1, 2, 3, 4, 8], ["0.25", "0.5", "1", "2", "3", "4", "8"]); ax[k].minorticks_off(); ax[k].legend(fontsize=7)
ax[0].set(ylabel="組がぴったり当たった割合 [%]", title="第 2 段階: 記号の表し方とずらしの量")
ax[1].set(ylabel="入れた光のうち答えに届いた割合 [%]", title="光をどれだけ捨てずに切り替えられたか")
fig.tight_layout(); fig.savefig("docs/images/exp16_relations_freq.png")
