"""実験111 の図: docs/figs/round_trip_111.svg（左: 並べ直しの当たり、右: 学んだ「前に来やすさ」）。実行: python tools/fig_round_trip_111.py"""
import json

STYLE = "<style>.ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
r = json.loads(open("results/exp111.txt", encoding="utf-8").read().strip().splitlines()[-1])
rows = [("でたらめ", r["random_exact"], 0.5, "#B8C7D9"), ("実験110 の逆回し", 0.16, 0.596, "#9FD3CC")] + [(x["name"], x["exact"], x["pairwise"], "#1F8F84") for x in r["runs"]]
b = txt("th", 230, 26, "混ぜた文節の並べ直し（3〜6 文節、300 文）")
b += txt("ts", 175, 50, "丸ごと一致") + txt("ts", 345, 50, "2 つずつの前後")
for i, (name, ex, pw, c) in enumerate(rows):
    y = 64 + i * 40
    b += txt("ts", 100, y + 14, name, "end")
    b += f"<rect x='110' y='{y}' width='{ex * 400}' height='18' fill='{c}' rx='2'/>" + txt("ts", 114 + ex * 400, y + 14, f"{ex:.0%}", "start")
    b += f"<rect x='280' y='{y}' width='{(pw - 0.4) * 400}' height='18' fill='{c}' rx='2'/>" + txt("ts", 284 + (pw - 0.4) * 400, y + 14, f"{pw:.0%}", "start")
b += txt("ts", 345, 64 + len(rows) * 40 + 10, "（2 つずつは 40% から描いている）")
x1 = 520; b += txt("th", 680, 26, "学んだ「前に来やすさ」（尾ごと）")
lead = r["lead"]; mx = max(abs(v) for v in lead.values())
for i, (t, v) in enumerate(lead.items()):
    y = 44 + i * 19; w = abs(v) / mx * 130
    x = 680 if v >= 0 else 680 - w
    b += f"<rect x='{x}' y='{y}' width='{w}' height='14' fill='{'#3B7DD8' if v >= 0 else '#D85A30'}' rx='2'/>" + txt("ts", 600 if v >= 0 else 700, y + 11, t, "end" if v >= 0 else "start")
b += txt("ts", 615, 44 + len(lead) * 19 + 14, "← 後ろ（文末）") + txt("ts", 745, 44 + len(lead) * 19 + 14, "前 →")
svg = f"<svg viewBox='0 0 880 360' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='880' height='360' fill='#fff'/>{b}</svg>"
open("docs/figs/round_trip_111.svg", "w", encoding="utf-8").write(svg)
print("ok")
