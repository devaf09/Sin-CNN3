"""「文法の逆回し」の使い道の図: docs/figs/reverse_use.svg（同じ重みを、解析の向きと生成の向きの両方に使う。往復で自分を確かめる）。
実行: python tools/fig_reverse_use.py"""
STYLE = "<style>.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222} .arr{stroke-width:1.8;fill:none}</style>"
DEFS = ("<defs><marker id='b' viewBox='0 0 10 10' refX='5' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#3B7DD8'/></marker>"
        "<marker id='o' viewBox='0 0 10 10' refX='5' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#D85A30'/></marker></defs>")
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
C = dict(blue=("#DCEBFA", "#3B7DD8"), coral=("#FCE3DA", "#D85A30"), green=("#DFF3DF", "#3C9A3C"), amber=("#FBEFD5", "#C98A12"), gray=("#EEEEEE", "#888"))


def box(x, y, w, h, col, lines):
    f, s = C[col]
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{f}' stroke='{s}'/>" + "".join(txt(c, x + w / 2, y + 19 + 17 * i, t) for i, (c, t) in enumerate(lines))


b = txt("th", 450, 24, "文法の逆回し: 同じ重み（共鳴 × 手がかりの重み θ）を、両方の向きに使う")
b += box(40, 60, 220, 70, "gray", [("th", "文（並んだ語）"), ("ts", "太郎は 浜辺で 亀を 見つけます")])
b += box(640, 60, 220, 70, "amber", [("th", "構造（重み・骨格）"), ("ts", "誰が・どこで・何を・見つける"), ("ts", "どの手がかりを重く見るか")])
b += box(340, 50, 220, 90, "green", [("th", "同じ重み"), ("ts", "語どうしの共鳴（PMI）"), ("ts", "× 手がかりの重み θ"), ("ts", "（穴埋めで学ぶ）")])
b += "<path class='arr' d='M260 80 L338 80' stroke='#3B7DD8' marker-end='url(#b)'/><path class='arr' d='M560 80 L638 80' stroke='#3B7DD8' marker-end='url(#b)'/>"
b += txt("ts", 450, 40, "解析の向き（読む）: 文 → 構造", "middle")
b += "<path class='arr' d='M638 115 L560 115' stroke='#D85A30' marker-end='url(#o)'/><path class='arr' d='M338 115 L262 115' stroke='#D85A30' marker-end='url(#o)'/>"
b += txt("ts", 450, 160, "生成の向き（書く）: 構造 → 文（穴埋め・並べ直し・骨格の展開）")
b += box(40, 190, 820, 110, "blue", [("th", "使い道"), ("ts", "1. 穴埋め = 1 語の生成（実験109・110）  2. 並べ直し = 語の順番の生成（実験110 の逆回し）  3. 骨格 → 文節の展開（実験105）"),
                                     ("ts", "4. 質問に答える = 物語の光の中の穴埋め（実験103）  5. 逆引き: 属性 → 語（語板の逆引き、実験76）"),
                                     ("ts", "6. 往復で自分を確かめる: 文 → 構造 → 文 と回して、元の文に戻るほど良い構造（笛のように吹いて返す）"),
                                     ("ts", "   答えが元の文そのものなので、自分の癖を強め合う輪（実験107・108）になりにくい")])
svg = f"<svg viewBox='0 0 900 320' xmlns='http://www.w3.org/2000/svg'>{STYLE}{DEFS}<rect width='900' height='320' fill='#fff'/>{b}</svg>"
open("docs/figs/reverse_use.svg", "w", encoding="utf-8").write(svg)
print("ok")
