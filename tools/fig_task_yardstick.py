"""「役割で測る」物差しの図: docs/figs/task_yardstick.svg（これまで: 木を GiNZA に近づける ／ これから: 構造は、当たるものが増えた時だけ残す）。
実行: python tools/fig_task_yardstick.py"""
STYLE = "<style>.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222} .arr{stroke:#666;stroke-width:1.4;fill:none}</style>"
DEFS = "<defs><marker id='a' viewBox='0 0 10 10' refX='5' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#666'/></marker></defs>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
C = dict(blue=("#DCEBFA", "#3B7DD8"), coral=("#FCE3DA", "#D85A30"), green=("#DFF3DF", "#3C9A3C"), gray=("#EEEEEE", "#888"), teal=("#D8F1EE", "#1F8F84"), amber=("#FBEFD5", "#C98A12"))


def box(x, y, w, h, col, lines):
    f, s = C[col]
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{f}' stroke='{s}'/>" + "".join(txt(c, x + w / 2, y + 19 + 17 * i, t) for i, (c, t) in enumerate(lines))


b = txt("th", 230, 24, "これまで（実験87〜108）") + txt("th", 690, 24, "これから")
b += box(30, 44, 180, 50, "blue", [("th", "材料（文）"), ("ts", "会話・青空文庫")]) + box(250, 44, 180, 50, "coral", [("th", "人が作った木"), ("ts", "GiNZA（正解として）")])
b += box(140, 140, 200, 54, "gray", [("th", "自分で張った木"), ("ts", "「GiNZA と同じか」で測る")])
b += "<path class='arr' d='M120 96 L200 138' marker-end='url(#a)'/><path class='arr' d='M340 96 L280 138' marker-end='url(#a)' stroke-dasharray='5 4'/>"
b += "<path class='arr' d='M140 180 C 60 200, 60 230, 150 230 C 230 230, 260 220, 240 196' marker-end='url(#a)'/>" + txt("ts", 120, 252, "自分の木で自分を学び直す → 癖を強め合う（67%）")
b += box(540, 44, 300, 50, "blue", [("th", "材料（文・会話・物語）")])
b += box(500, 130, 170, 64, "teal", [("th", "育つ構造"), ("ts", "木・メッシュ・重ね合わせ"), ("ts", "形は決めない")])
b += box(700, 130, 180, 64, "green", [("th", "当たるか（外の答え）"), ("ts", "次の語・穴埋め"), ("ts", "質問・話者関係")])
b += "<path class='arr' d='M640 96 L590 128' marker-end='url(#a)'/><path class='arr' d='M740 96 L790 128' marker-end='url(#a)'/><path class='arr' d='M672 162 L698 162' marker-end='url(#a)'/>"
b += "<path class='arr' d='M790 196 C 790 250, 590 250, 585 198' marker-end='url(#a)'/>" + txt("ts", 690, 262, "当たるものが増えた構造だけ残す（答えは材料そのもの）")
b += box(30, 290, 850, 74, "amber", [("th", "物差しの一覧（同じ材料・同じ試しの文で、構造だけ取り替えて比べる）"),
                                     ("ts", "1. 穴埋め: 文の中の語を隠して当てる（実験109）  2. 次の語・次の文の中身（実験77）  3. 浦島太郎の質問 30 問（実験87〜104）"),
                                     ("ts", "4. 話者関係: 誰が誰に話しているか・丁寧さから、文の結びを当てる（言結いの最初の目標、実験105 の ROOT 32%）")])
svg = f"<svg viewBox='0 0 900 380' xmlns='http://www.w3.org/2000/svg'>{STYLE}{DEFS}<rect width='900' height='380' fill='#fff'/>{b}</svg>"
open("docs/figs/task_yardstick.svg", "w", encoding="utf-8").write(svg)
print("ok")
