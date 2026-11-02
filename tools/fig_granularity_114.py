"""実験114 の図: docs/figs/granularity_114.svg（区切りの粒の長さと、1 文字あたりのビット。低いほど良く当たる）。実行: python tools/fig_granularity_114.py"""
import json, math

STYLE = "<style>.ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
R = json.loads(open("results/exp114.txt", encoding="utf-8").read().strip().splitlines()[-1])["runs"]
x0, y0, W, H = 80, 50, 560, 260
X = lambda L: x0 + (math.log2(L) / math.log2(13)) * W                                          # 単位の長さ（文字、対数）
Y = lambda b: y0 + H - (6.0 - b) / 4.2 * H                                                     # ビット（上ほど良い = 低い）
b = txt("th", 360, 26, "区切りの粒の長さと、次を当てる難しさ（1 文字あたりのビット、上ほど良い）")
for v in (2, 3, 4, 5, 6):
    b += f"<line x1='{x0}' x2='{x0 + W}' y1='{Y(v)}' y2='{Y(v)}' stroke='#eee'/>" + txt("ts", x0 - 6, Y(v) + 4, f"{v} ビット", "end")
for L in (1, 2, 4, 8, 12):
    b += txt("ts", X(L), y0 + H + 16, f"{L} 文字")
b += txt("ts", x0 + W / 2, y0 + H + 34, "1 つの単位の平均の長さ（右ほど長いかたまり）")
for lang, c in (("英語", "#3B7DD8"), ("日本語", "#D85A30")):
    rs = [r for r in R if r["lang"] == lang]
    learned = sorted([r for r in rs if not r["name"].startswith("答え")], key=lambda r: r["chars_per_unit"])
    b += f"<polyline points='{' '.join(f'{X(r['chars_per_unit']):.1f},{Y(r['bpc']):.1f}' for r in learned)}' fill='none' stroke='{c}' stroke-width='2'/>"
    for r in learned:
        b += f"<circle cx='{X(r['chars_per_unit']):.1f}' cy='{Y(r['bpc']):.1f}' r='3.5' fill='{c}'/>"
    g = next(r for r in rs if r["name"].startswith("答え"))
    b += f"<rect x='{X(g['chars_per_unit']) - 5:.1f}' y='{Y(g['bpc']) - 5:.1f}' width='10' height='10' fill='{c}'/>" + txt("ts", X(g["chars_per_unit"]) + 9, Y(g["bpc"]) + 4, f"{lang}: 答えの区切り {g['bpc']:.2f}", "start")
    lst = learned[-1]; b += txt("ts", X(lst["chars_per_unit"]) - 4, Y(lst["bpc"]) + (16 if lang == "英語" else -8), f"{lang}: 長いかたまり {lst['bpc']:.2f}", "end")
b += txt("ts", 360, y0 + H + 60, "● 学んだ区切り（区切りの数を変えた 5 段 + 文字 1 つずつ）  ■ 答えの区切り（英語 = 空白、日本語 = Sudachi の短い単位）")
svg = f"<svg viewBox='0 0 760 390' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='760' height='390' fill='#fff'/>{b}</svg>"
open("docs/figs/granularity_114.svg", "w", encoding="utf-8").write(svg)
print("ok")
