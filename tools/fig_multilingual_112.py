"""実験112 の図: docs/figs/round_trip_multilingual_112.svg（英語と日本語で同じ往復の並べ直し、覚える語の数ごと）。
文の例は Tatoeba（tatoeba.org、CC BY 2.0 FR）。実行: python tools/fig_multilingual_112.py"""
import json

STYLE = "<style>.ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
R = json.loads(open("results/exp112.txt", encoding="utf-8").read().strip().splitlines()[-1])["runs"]
b = txt("th", 460, 24, "同じ仕組みの往復（混ぜた語 → 並べ直し）を、英語と日本語で（4〜7 語の文 300）")
for gi, (metric, label, lo) in enumerate((("pairwise", "2 つずつの前後（でたらめ 50%）", 0.5), ("exact", "丸ごと一致（でたらめ 約 1%）", 0.0))):
    x0 = 60 + gi * 450; y0, H = 60, 200; top = 0.9 if metric == "pairwise" else 0.25
    b += txt("th", x0 + 180, 50, label)
    for v in ([0.5, 0.6, 0.7, 0.8, 0.9] if metric == "pairwise" else [0, 0.1, 0.2]):
        yy = y0 + H - (v - lo) / (top - lo) * H
        b += f"<line x1='{x0}' x2='{x0 + 370}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x0 - 6, yy + 4, f"{v:.0%}", "end")
    for li, (lang, c) in enumerate((("英語", "#3B7DD8"), ("日本語", "#D85A30"))):
        for ki, K in enumerate((50, 200, 1000)):
            r = next(x for x in R if x["lang"] == lang and x["K"] == K); v = r[metric]
            x = x0 + 20 + ki * 120 + li * 46; h = (v - lo) / (top - lo) * H
            b += f"<rect x='{x}' y='{y0 + H - h}' width='40' height='{h}' fill='{c}' rx='2'/>" + txt("ts", x + 20, y0 + H - h - 4, f"{v:.0%}")
        b += f"<rect x='{x0 + 100 + li * 100}' y='{y0 + H + 34}' width='12' height='12' fill='{c}'/>" + txt("ts", x0 + 116 + li * 100, y0 + H + 45, lang, "start")
    for ki, K in enumerate((50, 200, 1000)):
        b += txt("ts", x0 + 63 + ki * 120, y0 + H + 16, f"覚える語 {K}")
b += txt("ts", 460, 330, "言語の知識は入れていない（よく出る語そのもの + まれさの帯だけ）。語の切り方だけは言語ごと。文は Tatoeba（CC BY 2.0 FR）")
svg = f"<svg viewBox='0 0 940 345' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='940' height='345' fill='#fff'/>{b}</svg>"
open("docs/figs/round_trip_multilingual_112.svg", "w", encoding="utf-8").write(svg)
print("ok")
