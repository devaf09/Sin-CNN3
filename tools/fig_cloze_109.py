"""実験109 の図: docs/figs/structure_by_cloze_109.svg（構造ごとの穴埋め MRR と、その木の係り先の正答）。実行: python tools/fig_cloze_109.py"""
import json

STYLE = "<style>.ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222} .ax{stroke:#999;stroke-width:1}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
r = json.loads(open("results/exp109.txt", encoding="utf-8").read().strip().splitlines()[-1])
x0, y0, H, top = 60, 50, 220, 0.08
b = txt("th", 330, 26, "構造ごとの穴埋めの当たり（MRR、試し 6600 問）")
for v in (0, 0.02, 0.04, 0.06, 0.08):
    yy = y0 + H - v / top * H
    b += f"<line x1='{x0}' x2='{x0 + 540}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x0 - 6, yy + 4, f"{v:.2f}", "end")
labels = {"袋": ("袋", "（全部同じ）"), "並びの近さ": ("並びの近さ", ""), "GiNZA の木": ("GiNZA の木", "係り先 100%"),
          "自分の木（答えなし）": ("自分の木", f"答えなし {r['uas']['自分の木（答えなし）']:.0%}"), "自分の木（答えあり + 遠さ）": ("自分の木", f"答えあり {r['uas']['自分の木（答えあり + 遠さ）']:.0%}")}
for i, run in enumerate(r["runs"]):
    v = run["mrr"]; x = x0 + 20 + i * 106; h = v / top * H; c = "#D85A30" if "GiNZA" in run["name"] else ("#1F8F84" if "自分" in run["name"] else "#B8C7D9")
    b += f"<rect x='{x}' y='{y0 + H - h}' width='66' height='{h}' fill='{c}' rx='2'/>" + txt("ts", x + 33, y0 + H - h - 5, f"{v:.4f}")
    b += txt("ts", x + 33, y0 + H + 16, labels[run["name"]][0]) + txt("ts", x + 33, y0 + H + 31, labels[run["name"]][1])
b += txt("ts", 330, y0 + H + 56, "正しい木は穴埋めを 4 割以上よくする。自分の木（77%）は並びの近さと同じ = 外した係り先が、意味の手がかりの所")
svg = f"<svg viewBox='0 0 660 340' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='660' height='340' fill='#fff'/>{b}</svg>"
open("docs/figs/structure_by_cloze_109.svg", "w", encoding="utf-8").write(svg)
print("ok")
