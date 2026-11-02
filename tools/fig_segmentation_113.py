"""実験113 の図: docs/figs/segmentation_113.svg（文字の並びだけから学んだ区切りの F1、英語・日本語）。文の例は Tatoeba（CC BY 2.0 FR）。
実行: python tools/fig_segmentation_113.py"""
import json

STYLE = "<style>.ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222} .t{font:13px monospace;fill:#222}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
R = json.loads(open("results/exp113.txt", encoding="utf-8").read().strip().splitlines()[-1])
langs = ["英語", "日本語（短い単位）", "日本語（文節）"]
b = txt("th", 470, 24, "文字の並びだけから学んだ語の区切り（区切りの F1、試し 2000 文）")
x0, y0, H = 70, 50, 200
for v in (0, 0.2, 0.4, 0.6, 0.8):
    yy = y0 + H - v * H
    b += f"<line x1='{x0}' x2='{x0 + 560}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x0 - 6, yy + 4, f"{v:.1f}", "end")
for gi, lang in enumerate(langs):
    rs = [x for x in R["runs"] if x["lang"] == lang]
    best = max((x for x in rs if isinstance(x["rate"], float)), key=lambda x: x["f1"])
    items = [("学んだ", best["f1"], "#1F8F84"), ("でたらめ", next(x["f1"] for x in rs if x["rate"] == "でたらめ"), "#B8C7D9"), ("毎文字", next(x["f1"] for x in rs if x["rate"] == "毎文字"), "#D9D9D9")]
    for i, (n, v, c) in enumerate(items):
        x = x0 + 20 + gi * 190 + i * 52; h = v * H
        b += f"<rect x='{x}' y='{y0 + H - h}' width='44' height='{h}' fill='{c}' rx='2'/>" + txt("ts", x + 22, y0 + H - h - 4, f"{v:.2f}") + txt("ts", x + 22, y0 + H + 14, n)
    b += txt("th", x0 + 98 + gi * 190, y0 + H + 34, lang)
y = 320; b += txt("th", 30, y, "例（上: 答え、下: 学んだ区切り）", "start")
for lang in ("英語", "日本語（短い単位）"):
    for e in R["examples"][lang][1:3]:
        y += 22; b += txt("t", 40, y, e["answer"], "start"); y += 18; b += txt("t", 40, y, e["learned"], "start")
    y += 6
b += txt("ts", 470, y + 24, "文は Tatoeba（tatoeba.org、CC BY 2.0 FR）。英語は空白を消して学び、元の空白で答え合わせ。日本語は Sudachi の短い単位と GiNZA の文節で答え合わせ")
svg = f"<svg viewBox='0 0 940 {y + 40}' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='940' height='{y + 40}' fill='#fff'/>{b}</svg>"
open("docs/figs/segmentation_113.svg", "w", encoding="utf-8").write(svg)
print("ok")
