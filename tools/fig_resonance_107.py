"""実験107（共鳴パーシング）の図: docs/figs/resonance_parser_107.svg
左: 係り先の正答（GiNZA と同じ所に矢印を張れた割合）。右: 答えなしの自己学習（自分の木で数え直す）の回数ごとの正答。
実行: python tools/fig_resonance_107.py（results/exp107.txt の最後の行を読む）"""
import json

STYLE = "<style>.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222} .ax{stroke:#999;stroke-width:1}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
R = json.loads(open("results/exp107.txt", encoding="utf-8").read().strip().splitlines()[-1])["runs"]
get = lambda name, **kw: next(r for r in R if r["name"] == name and all(r.get(k) == v for k, v in kw.items()))
bars = [("最後の文節", get("最後の文節")["uas"], "#B8C7D9"), ("すぐ右", get("すぐ右")["uas"], "#B8C7D9"),
        ("答えなし（自己学習）", get("答えなし（品詞だけ・割合、すぐ右から自分の木で数え直し）", sat=0)["uas"], "#1F8F84"),
        ("答えあり（共鳴）", get("答えあり（品詞だけ・割合）", sat=0)["uas"], "#3B7DD8"), ("手の決まり", get("手の決まり")["uas"], "#D85A30")]
x0, y0, H = 60, 50, 220; lo = 0.4
b = txt("th", 270, 26, "係り先が GiNZA と同じ割合（試し 1500 文）")
b += f"<line class='ax' x1='{x0}' y1='{y0 + H}' x2='{x0 + 430}' y2='{y0 + H}'/>"
for v in (0.4, 0.5, 0.6, 0.7, 0.8):
    yy = y0 + H - (v - lo) / 0.4 * H
    b += f"<line x1='{x0}' x2='{x0 + 430}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x0 - 6, yy + 4, f"{v:.0%}", "end")
for i, (name, v, c) in enumerate(bars):
    x = x0 + 14 + i * 84; h = (v - lo) / 0.4 * H
    b += f"<rect x='{x}' y='{y0 + H - h}' width='56' height='{h}' fill='{c}' rx='2'/>" + txt("ts", x + 28, y0 + H - h - 5, f"{v:.1%}")
    for k, part in enumerate(name.replace("（", "|（").split("|")):
        b += txt("ts", x + 28, y0 + H + 16 + 15 * k, part)
cur = get("答えなし（品詞だけ・割合、すぐ右から自分の木で数え直し）", sat=0)["curve"]
x1 = 580; b += txt("th", 750, 26, "答えなしで自分の木から学び直す")
b += f"<line class='ax' x1='{x1}' y1='{y0 + H}' x2='{x1 + 330}' y2='{y0 + H}'/>"
for v in (0.6, 0.65, 0.7, 0.75):
    yy = y0 + H - (v - 0.6) / 0.17 * H
    b += f"<line x1='{x1}' x2='{x1 + 330}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x1 - 6, yy + 4, f"{v:.0%}", "end")
pts = " ".join(f"{x1 + 20 + i * 58},{y0 + H - (v - 0.6) / 0.17 * H:.1f}" for i, v in enumerate(cur))
b += f"<polyline points='{pts}' fill='none' stroke='#1F8F84' stroke-width='2'/>"
for i, v in enumerate(cur):
    b += f"<circle cx='{x1 + 20 + i * 58}' cy='{y0 + H - (v - 0.6) / 0.17 * H:.1f}' r='3' fill='#1F8F84'/>" + txt("ts", x1 + 20 + i * 58, y0 + H + 16, f"{i} 回")
for v, name, c in ((get("すぐ右")["uas"], "すぐ右", "#888"), (get("手の決まり")["uas"], "手の決まり", "#D85A30")):
    yy = y0 + H - (v - 0.6) / 0.17 * H
    b += f"<line x1='{x1}' x2='{x1 + 330}' y1='{yy}' y2='{yy}' stroke='{c}' stroke-dasharray='5 4'/>" + txt("ts", x1 + 330, yy - 4, name, "end")
b += txt("ts", 745, y0 + H + 40, "すぐ右の癖を自分で強め合い、そこから抜け出せない")
svg = f"<svg viewBox='0 0 960 340' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='960' height='340' fill='#fff'/>{b}</svg>"
open("docs/figs/resonance_parser_107.svg", "w", encoding="utf-8").write(svg)
print("ok")
