"""実験100（語板 v5 を質問の重みとバッファの変動に使う）の図: docs/figs/v5_question_100.svg
左: 4 通りの正答（浦島太郎 30 問）。右: 動詞の語板 v5 が予想する「答えになりそうな名詞の仲間」（持ち上がりの上位）。
実行: python tools/fig_v5_100.py（results/exp100.txt の最後の行を読む）"""
import json

STYLE = "<style>.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222} .ax{stroke:#999;stroke-width:1}</style>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
r = json.loads(open("results/exp100.txt", encoding="utf-8").read().strip().splitlines()[-1])
modes = [k for k in r if k not in ("exp", "examples")]
x0, y0, H = 60, 50, 220
b = txt("th", 230, 24, "浦島太郎 30 問の正答")
b += f"<line class='ax' x1='{x0}' y1='{y0 + H}' x2='{x0 + 340}' y2='{y0 + H}'/>"
for v in (0, 10, 20, 30):
    yy = y0 + H - v / 30 * H
    b += f"<line x1='{x0}' x2='{x0 + 340}' y1='{yy}' y2='{yy}' stroke='#eee'/>" + txt("ts", x0 - 6, yy + 4, str(v), "end")
labels = [("名札で", "動詞 2 倍"), ("v5 の品詞", "の光で重み"), ("+ 答えの", "種類 β=0.3"), ("+ 答えの", "種類 β=0.6")]
for i, m in enumerate(modes):
    v = r[m]["q20"] + r[m]["q10"]; x = x0 + 20 + i * 82
    b += f"<rect x='{x}' y='{y0 + H - v / 30 * H}' width='50' height='{v / 30 * H}' fill='{'#3B7DD8' if i < 2 else '#D85A30'}' rx='2'/>" + txt("ts", x + 25, y0 + H - v / 30 * H - 4, str(v))
    b += txt("ts", x + 25, y0 + H + 16, labels[i][0]) + txt("ts", x + 25, y0 + H + 32, labels[i][1])
b += txt("th", 690, 24, "動詞の語板 v5 が予想する答えの仲間（持ち上がり）")
y = 56
for k, rows in r["examples"].items():
    b += txt("th", 470, y, k, "start"); y += 20
    for name, lift in rows:
        b += f"<rect x='480' y='{y - 11}' width='{min(lift, 160) * 1.2}' height='12' fill='#1F8F84' opacity='0.5'/>" + txt("ts", 486, y, f"{name}（×{lift}）", "start"); y += 18
    y += 8
svg = f"<svg viewBox='0 0 960 {max(y, 340) + 10}' xmlns='http://www.w3.org/2000/svg'>{STYLE}<rect width='960' height='{max(y, 340) + 10}' fill='#fff'/>{b}</svg>"
open("docs/figs/v5_question_100.svg", "w", encoding="utf-8").write(svg)
print("ok")
