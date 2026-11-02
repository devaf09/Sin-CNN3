"""実験115 の図: docs/figs/response_selection_115.svg（10 択の返事選びの 1 位の正答。ほかの会話から / 同じ会話から）。実行: python tools/fig_response_115.py"""
import json

STYLE = "<style>.ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
R = json.loads(open("results/exp115.txt", encoding="utf-8").read().strip().splitlines()[-1])["runs"]
names = ["でたらめ", "語の重なり", "次の発言の共鳴", "重なり + 共鳴", "+ 文脈 3 発言", "+ 口調の合い"]
b = txt("th", 400, 24, "返事を 10 個の候補から選ぶ（1 位の正答、試し 3000 問）")
x0, y0, H = 70, 60, 220
for v in (0, 0.2, 0.4, 0.6):
    yy = y0 + H - v / 0.7 * H
    b += f"<line x1='{x0}' x2='{x0 + 680}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x0 - 6, yy + 4, f"{v:.0%}", "end")
for i, n in enumerate(names):
    for j, (hard, c) in enumerate(((False, "#3B7DD8"), (True, "#D85A30"))):
        r = next(x for x in R if x["name"] == n and x["hard"] == hard); v = r["top1"]
        x = x0 + 20 + i * 112 + j * 44; h = v / 0.7 * H
        b += f"<rect x='{x}' y='{y0 + H - h}' width='40' height='{h}' fill='{c}' rx='2'/>" + txt("ts", x + 20, y0 + H - h - 4, f"{v:.0%}")
    b += txt("ts", x0 + 62 + i * 112, y0 + H + 16, n)
b += f"<rect x='200' y='{y0 + H + 34}' width='12' height='12' fill='#3B7DD8'/>" + txt("ts", 218, y0 + H + 45, "外れの候補 = ほかの会話の発言（話題で見分けられる）", "start")
b += f"<rect x='200' y='{y0 + H + 54}' width='12' height='12' fill='#D85A30'/>" + txt("ts", 218, y0 + H + 65, "外れの候補 = 同じ会話のほかの発言（流れで見分ける、むずかしい）", "start")
svg = f"<svg viewBox='0 0 800 360' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='800' height='360' fill='#fff'/>{b}</svg>"
open("docs/figs/response_selection_115.svg", "w", encoding="utf-8").write(svg)
print("ok")
