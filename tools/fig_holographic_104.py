"""実験104 の図: docs/figs/holographic_select_104.svg
左: 「まわり」の考え（選んだ枠から離れるほど γ^距離 で薄く足す）。右: 8 通りの質問の正答（D = 16384）。
実行: python tools/fig_holographic_104.py（results/exp104.txt の最後の行を読む）"""
import json

STYLE = """<style>.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}
.arr{stroke:#666;stroke-width:1.4} .ax{stroke:#999;stroke-width:1}</style>"""
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
runs = [r for r in json.loads(open("results/exp104.txt", encoding="utf-8").read().strip().splitlines()[-1])["runs"] if r["D"] == 16384]
b = txt("th", 190, 24, "まわり: 離れた枠ほど薄く足す（γ = 0.5）")
# 木の上の枠（同じ文）: 中心の枠と、距離 1・2 の枠
nodes = [(190, 80, "選んだ枠", 1.0), (90, 160, "距離 1", 0.5), (290, 160, "距離 1", 0.5), (50, 240, "距離 2", 0.25), (140, 240, "距離 2", 0.25), (330, 240, "距離 2", 0.25)]
for (x1, y1), (x2, y2) in (((190, 80), (90, 160)), ((190, 80), (290, 160)), ((90, 160), (50, 240)), ((90, 160), (140, 240)), ((290, 160), (330, 240))):
    b += f"<line class='arr' x1='{x1}' y1='{y1 + 18}' x2='{x2}' y2='{y2 - 18}'/>"
for x, y, t, w in nodes:
    b += f"<rect x='{x - 42}' y='{y - 18}' width='84' height='36' rx='8' fill='#3B7DD8' fill-opacity='{0.15 + 0.6 * w}' stroke='#3B7DD8'/>" + txt("ts", x, y - 2, t) + txt("ts", x, y + 13, f"× {w}")
b += txt("ts", 190, 290, "枠の点 = 自分の重なり + 0.5 × 隣 + 0.25 × その隣 …")
b += txt("ts", 190, 308, "答えも、まわりの枠から 0.5^距離 を掛けて探す")
b += txt("ts", 190, 336, "足し算: Σ 重み × 重なり（どれかが合えばよい）")
b += txt("ts", 190, 354, "掛け算: Π 重なり^重み（全部揃う枠だけ = AND）")
# 右: 棒
x0, y0, H = 440, 60, 260
b += txt("th", 680, 24, "質問 30 問の正答（D = 16384）")
b += f"<line class='ax' x1='{x0}' y1='{y0 + H}' x2='{x0 + 470}' y2='{y0 + H}'/>"
for v in (0, 10, 20, 30):
    yy = y0 + H - v / 30 * H
    b += f"<line x1='{x0}' x2='{x0 + 470}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x0 - 6, yy + 4, str(v), "end")
yy = y0 + H - 29 / 30 * H
b += f"<line x1='{x0}' x2='{x0 + 470}' y1='{yy}' y2='{yy}' stroke='#D85A30' stroke-dasharray='5 4'/>" + txt("ts", x0 + 470, yy - 6, "文章樹 29", "end")
for i, r in enumerate(runs):
    v = r["q20"] + r["q10"]; x = x0 + 12 + i * 57; c = "#1F8F84" if r["ans_near"] else "#9FD3CC"
    b += f"<rect x='{x}' y='{y0 + H - v / 30 * H}' width='40' height='{v / 30 * H}' fill='{c}' rx='2'/>" + txt("ts", x + 20, y0 + H - v / 30 * H - 4, str(v))
    b += txt("ts", x + 20, y0 + H + 16, "まわり" if r["near"] else "なし") + txt("ts", x + 20, y0 + H + 32, r["comb"])
b += txt("ts", x0 - 6, y0 + H + 16, "選ぶ:", "end") + txt("ts", x0 - 6, y0 + H + 32, "点:", "end")
b += f"<rect x='{x0 + 60}' y='{y0 + H + 46}' width='14' height='14' fill='#9FD3CC'/>" + txt("ts", x0 + 80, y0 + H + 58, "答えは選んだ枠だけ", "start")
b += f"<rect x='{x0 + 240}' y='{y0 + H + 46}' width='14' height='14' fill='#1F8F84'/>" + txt("ts", x0 + 260, y0 + H + 58, "答えはまわりの枠も", "start")
svg = f"<svg viewBox='0 0 940 400' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='940' height='400' fill='#fff'/>{b}</svg>"
open("docs/figs/holographic_select_104.svg", "w", encoding="utf-8").write(svg)
print("ok")
