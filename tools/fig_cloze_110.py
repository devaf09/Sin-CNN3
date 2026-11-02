"""実験110 の図: docs/figs/cloze_grown_110.svg（左: 同じ試しの穴埋め MRR、右: 逆回しの並べ直し）。実行: python tools/fig_cloze_110.py"""
import json

STYLE = "<style>.ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
L = [json.loads(l) for l in open("results/exp110.txt", encoding="utf-8")]
main = [x for x in L if x["exp"] == "110"][-1]; fx = [x for x in L if x["exp"] == "110b"][-1]
bars = [("学ぶ前", "（袋）", 0.0547, "#B8C7D9"), ("並びの", "近さ", fx["並びの近さ"], "#B8C7D9"), ("穴埋めで", "学んだ重み", fx["学んだ重み_最良"], "#1F8F84"), ("GiNZA", "の木", fx["GiNZAの木"], "#D85A30")]
x0, y0, H, top = 60, 50, 220, 0.09
b = txt("th", 250, 26, "穴埋め MRR（試し 6796 問）")
for v in (0, 0.03, 0.06, 0.09):
    yy = y0 + H - v / top * H
    b += f"<line x1='{x0}' x2='{x0 + 380}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x0 - 6, yy + 4, f"{v:.2f}", "end")
for i, (a, c2, v, c) in enumerate(bars):
    x = x0 + 20 + i * 92; h = v / top * H
    b += f"<rect x='{x}' y='{y0 + H - h}' width='60' height='{h}' fill='{c}' rx='2'/>" + txt("ts", x + 30, y0 + H - h - 5, f"{v:.4f}") + txt("ts", x + 30, y0 + H + 16, a) + txt("ts", x + 30, y0 + H + 31, c2)
b += txt("ts", 250, y0 + H + 56, "人の木を使わずに、並び → 人の木 の差の約 45% を埋めた")
r = main["reorder"]; x1 = 520
b += txt("th", 690, 26, "逆回し: 混ぜた文節の並べ直し（3〜6 文節、300 文）")
for i, (name, v, rv) in enumerate((("丸ごと一致", r["exact"], r["exact_random"]), ("2 つずつの前後", r["pairwise"], 0.5))):
    y = 70 + i * 90
    b += txt("ts", x1, y, name, "start")
    b += f"<rect x='{x1}' y='{y + 8}' width='{v * 300}' height='18' fill='#1F8F84' rx='2'/>" + txt("ts", x1 + v * 300 + 6, y + 22, f"学んだ重み {v:.0%}", "start")
    b += f"<rect x='{x1}' y='{y + 32}' width='{rv * 300}' height='18' fill='#B8C7D9' rx='2'/>" + txt("ts", x1 + rv * 300 + 6, y + 46, f"でたらめ {rv:.0%}", "start")
b += txt("ts", 690, 260, "同じ重みを逆に回すと、並びも少し当たる（まだ弱い）")
svg = f"<svg viewBox='0 0 920 340' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='920' height='340' fill='#fff'/>{b}</svg>"
open("docs/figs/cloze_grown_110.svg", "w", encoding="utf-8").write(svg)
print("ok")
