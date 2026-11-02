"""実験108 の図: docs/figs/resonance_parser_108.svg（答えの量ごとに、自己学習の回数で正答がどう動くか）。実行: python tools/fig_resonance_108.py"""
import json

STYLE = "<style>.ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222} .ax{stroke:#999;stroke-width:1}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
R = json.loads(open("results/exp108.txt", encoding="utf-8").read().strip().splitlines()[-1])["runs"]
x0, y0, W, H, lo, hi = 70, 50, 420, 240, 0.64, 0.78
Y = lambda v: y0 + H - (v - lo) / (hi - lo) * H
b = txt("th", 330, 26, "自分の解析で数え直す回数と、係り先の正答（実験108）")
for v in (0.64, 0.68, 0.72, 0.76):
    b += f"<line x1='{x0}' x2='{x0 + W}' y1='{Y(v)}' y2='{Y(v)}' stroke='#eee'/>" + txt("ts", x0 - 6, Y(v) + 4, f"{v:.0%}", "end")
cols = {"答えなし（硬い自己学習、共鳴 + 遠さ）": ("#888", "答えなし"), "答え 100 文 + 残りは自己学習": ("#9FD3CC", "答え 100 文"),
        "答え 300 文 + 残りは自己学習": ("#1F8F84", "答え 300 文"), "答え 1000 文 + 残りは自己学習": ("#3B7DD8", "答え 1000 文")}
for r in R:
    if r["name"] in cols:
        c, lab = cols[r["name"]]; cur = r["curve"]
        pts = " ".join(f"{x0 + 20 + i * 80},{Y(v):.1f}" for i, v in enumerate(cur))
        b += f"<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='2'/>" + txt("ts", x0 + 26 + (len(cur) - 1) * 80, Y(cur[-1]) + 4, lab, "start")
for i in range(6):
    b += txt("ts", x0 + 20 + i * 80, y0 + H + 16, f"{i} 回")
for name, lab, c in (("答えあり 共鳴 + 遠さ", "答えすべて（1.3 万文）+ 遠さ 76.6%", "#D85A30"),):
    v = next(r["uas"] for r in R if r["name"] == name)
    b += f"<line x1='{x0}' x2='{x0 + W}' y1='{Y(v)}' y2='{Y(v)}' stroke='{c}' stroke-dasharray='5 4'/>" + txt("ts", x0 + W, Y(v) - 5, lab, "end")
b += txt("ts", 330, y0 + H + 40, "自分の解析で数え直すほど、答えの量によらず下がる（自分の癖を強め合う）")
svg = f"<svg viewBox='0 0 660 340' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='660' height='340' fill='#fff'/>{b}</svg>"
open("docs/figs/resonance_parser_108.svg", "w", encoding="utf-8").write(svg)
print("ok")
