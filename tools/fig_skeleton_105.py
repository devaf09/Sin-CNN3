"""実験105（骨格 → 展開）の図: docs/figs/skeleton_expand_105.svg
上: 2 段の流れ（骨格のスロット → 各スロットを並列に文節へ）。下: 役割ごとの文節の正答（比べ 0・3・5、1 語ずつ、試しの文節）。
実行: python tools/fig_skeleton_105.py（results/exp105.txt の最後の行を読む）"""
import json

STYLE = """<style>.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}
.arr{stroke:#666;stroke-width:1.4} .ax{stroke:#999;stroke-width:1}</style>"""
DEFS = "<defs><marker id='a' viewBox='0 0 10 10' refX='5' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#666'/></marker></defs>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"


def box(x, y, w, h, fill, stroke, lines):
    s = f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{fill}' stroke='{stroke}'/>"
    return s + "".join(txt(c, x + w / 2, y + 19 + 17 * i, t) for i, (c, t) in enumerate(lines))


runs = json.loads(open("results/exp105.txt", encoding="utf-8").read().strip().splitlines()[-1])["runs"]
b = txt("th", 470, 22, "1 段目: 骨格（抽象的なスロット）→ 2 段目: スロットごとに並列で文節へ展開")
slots = [("nsubj", "太郎", "名詞"), ("obl", "浜辺", "名詞"), ("obj", "亀", "名詞"), ("ROOT", "見付ける", "動詞・丁寧")]
outs = ["太郎は", "浜辺で", "亀を", "見つけます。"]
for i, ((d, w, p), o) in enumerate(zip(slots, outs)):
    x = 30 + i * 230
    b += box(x, 40, 200, 58, "#DCEBFA", "#3B7DD8", [("th", f"{d} ・ {w}"), ("ts", p)])
    b += f"<path class='arr' d='M{x + 100} 100 L{x + 100} 140' marker-end='url(#a)'/>"
    b += box(x, 144, 200, 40, "#DFF3DF", "#3C9A3C", [("t", o)])
b += f"<rect x='30' y='112' width='890' height='18' fill='#FBEFD5' opacity='0.8'/>" + txt("ts", 475, 125, "参照光: 係る先の述語の語（見付ける）を全スロットへ配る  ／  格まで骨格が決める版: nsubj:は・obl:で・obj:を")
# 下: 役割ごとの棒
x0, y0, H = 70, 230, 170
deps = ["obj", "nmod", "advmod", "nsubj", "acl", "obl", "ROOT", "advcl"]
R = {r["model"]: r["試し"]["by_dep"] for r in runs if r["how"] == "1 語ずつ"}
cols = [(0, "#B8C7D9", "役割・品詞・印だけ"), (3, "#3B7DD8", "+ 中身の語 + 参照光"), (5, "#D85A30", "+ 格まで骨格が決める")]
b += txt("th", 470, y0 - 12, "役割ごとの文節の正答（試しの文節 2.2 万、1 語ずつの展開）")
b += f"<line class='ax' x1='{x0}' y1='{y0 + H}' x2='{x0 + 830}' y2='{y0 + H}'/>"
for v in (0, 0.5, 1):
    yy = y0 + H - v * H
    b += f"<line x1='{x0}' x2='{x0 + 830}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x0 - 6, yy + 4, f"{int(v * 100)}%", "end")
for i, d in enumerate(deps):
    gx = x0 + 14 + i * 103
    for j, (m, c, _) in enumerate(cols):
        v = R[m].get(d, 0); x = gx + j * 27
        b += f"<rect x='{x}' y='{y0 + H - v * H}' width='24' height='{v * H}' fill='{c}' rx='2'/>"
    b += txt("ts", gx + 40, y0 + H + 16, d)
for j, (_, c, name) in enumerate(cols):
    b += f"<rect x='{x0 + 40 + j * 250}' y='{y0 + H + 30}' width='14' height='14' fill='{c}'/>" + txt("ts", x0 + 60 + j * 250, y0 + H + 42, name, "start")
b += txt("ts", 470, y0 + H + 68, "全体: 47% → 49% → 60%（句読点を除いた尾は 52% → 55% → 68%）。決まらないのは 文の結び（ROOT）と つなぎ（advcl）")
svg = f"<svg viewBox='0 0 950 500' xmlns='http://www.w3.org/2000/svg'>{STYLE}{DEFS}<rect width='950' height='500' fill='#fff'/>{b}</svg>"
open("docs/figs/skeleton_expand_105.svg", "w", encoding="utf-8").write(svg)
print("ok")
