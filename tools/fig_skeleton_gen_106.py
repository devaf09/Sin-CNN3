"""実験106（骨格生成器）の図: docs/figs/skeleton_generator_106.svg
左: 試しの述語の格の組（半分は格なし）。中: いちばん要求の強い格が当たる割合（A〜D）。右: よく出る述語の要求（格の割合）。
実行: python tools/fig_skeleton_gen_106.py"""
import json

STYLE = "<style>.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222} .ax{stroke:#999;stroke-width:1}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
L = [json.loads(l) for l in open("results/exp106.txt", encoding="utf-8").read().strip().splitlines()]
main = [x for x in L if x["exp"] == "106"][-1]; extra = [x for x in L if x["exp"] == "106b"][-1]
b = txt("th", 140, 24, "試しの述語の格の組")
tot = main["n_test"]; y = 44
for s, n in main["set_dist"][:8]:
    w = n / tot * 400
    b += f"<rect x='90' y='{y}' width='{w}' height='16' fill='#3B7DD8' rx='2'/>" + txt("ts", 84, y + 12, "・".join(s) or "（格なし）", "end") + txt("ts", 96 + w, y + 12, f"{n / tot:.0%}", "start"); y += 22
b += txt("ts", 140, y + 14, "述語の半分は、格を 1 つも言わない（省略）")
x0, y0, H = 330, 50, 180
b += txt("th", 450, 24, "いちばん要求の強い格が当たる割合")
b += f"<line class='ax' x1='{x0}' y1='{y0 + H}' x2='{x0 + 250}' y2='{y0 + H}'/>"
names = dict(A=("事前", ""), B=("述語の", "要求"), C=("+ 係り方", ""), D=("+ v5 で", "借りる"))
for i, k in enumerate("ABCD"):
    v = extra["top_case"][k]["all"]; x = x0 + 10 + i * 60
    b += f"<rect x='{x}' y='{y0 + H - v * H}' width='40' height='{v * H}' fill='{'#B8C7D9' if k == 'A' else '#1F8F84'}' rx='2'/>" + txt("ts", x + 20, y0 + H - v * H - 4, f"{v:.0%}") + txt("ts", x + 20, y0 + H + 16, names[k][0]) + txt("ts", x + 20, y0 + H + 31, names[k][1])
b += txt("ts", x0 + 125, y0 + H + 52, f"格が出た述語 {extra['n_withslot']} 個で。教えに少ない述語では v5 で借りると下がる（42% → 35%）")
x1 = 640; b += txt("th", 820, 24, "述語の要求（その格が出る割合）")
y = 44
for v, fr in list(main["frames"].items())[:9]:
    b += txt("ts", x1 + 40, y + 12, v, "end"); xx = x1 + 46
    for c, p in sorted(fr.items(), key=lambda t: -t[1])[:3]:
        w = max(p * 260, 30)
        b += f"<rect x='{xx}' y='{y}' width='{w}' height='16' fill='#D85A30' opacity='{0.4 + 0.6 * p}' rx='2'/>" + txt("ts", xx + 3, y + 12, f"{c} {p:.0%}", "start"); xx += w + 2
    y += 22
b += txt("ts", 820, y + 14, "要求は確率: 食べる の を は 17% だけ")
svg = f"<svg viewBox='0 0 1060 330' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='1060' height='330' fill='#fff'/>{b}</svg>"
open("docs/figs/skeleton_generator_106.svg", "w", encoding="utf-8").write(svg)
print("ok")
