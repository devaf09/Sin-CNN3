"""実験120 の図: docs/figs/transition_grain_120.svg（ラベルなしでできた発言の状態と、よく起きる遷移）。名前は中身を見て後から付けたもの。
実行: python tools/fig_transition_120.py"""
import json

STYLE = "<style>.ts{font:12px sans-serif;fill:#555} .th{font:bold 13px sans-serif;fill:#222} .arr{stroke:#666;fill:none}</style>"
DEFS = "<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#666'/></marker></defs>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
r = json.loads(open("results/exp120.txt", encoding="utf-8").read().strip().splitlines()[-1])
NAME = {13: "あいさつ", 7: "問い（〜か？）", 11: "問い（感想 + 問い）", 2: "相手への問いかけ", 12: "答え・話の続き", 1: "共感（〜ですね。）", 4: "自分の話", 10: "話題を振る（〜ですね）", 14: "あいづち → 切り替え", 6: "受けて自分の側へ"}
POS = {13: (110, 80), 7: (330, 80), 11: (330, 200), 2: (110, 200), 12: (560, 140), 1: (790, 80), 4: (110, 320), 10: (790, 200), 14: (560, 320), 6: (790, 320)}
b = txt("th", 450, 24, "ラベルなしでできた発言の状態と、よく起きる遷移（実験120、名前は中身を見て後から付けた）")
TT = r["top_transitions"]
for k, (x, y) in POS.items():
    for j, p in TT[str(k)][:2]:
        if j in POS and j != k:
            x2, y2 = POS[j]
            b += f"<path class='arr' d='M{x} {y} L{x2} {y2}' stroke-width='{1 + 8 * p}' stroke-opacity='0.6' marker-end='url(#a)'/>"
            b += txt("ts", (x + x2) / 2, (y + y2) / 2 - 4, f"{p:.2f}")
for k, (x, y) in POS.items():
    st = r["states"][str(k)]
    b += f"<rect x='{x - 85}' y='{y - 22}' width='170' height='44' rx='8' fill='#DCEBFA' stroke='#3B7DD8'/>" + txt("th", x, y - 3, NAME[k]) + txt("ts", x, y + 14, f"発言の {st['share'] * 100:.0f}%")
b += txt("ts", 450, 380, f"同じ会話の 10 択: 重なり + 共鳴 {r['base']:.0%}、遷移だけ {r['transition_only']:.0%}（でたらめ 10%）、両方 {r['base_plus_transition']:.0%}")
svg = f"<svg viewBox='0 0 900 400' xmlns='http://www.w3.org/2000/svg'>{STYLE}{DEFS}<rect width='900' height='400' fill='#fff'/>{b}</svg>"
open("docs/figs/transition_grain_120.svg", "w", encoding="utf-8").write(svg)
print("ok")
