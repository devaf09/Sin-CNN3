"""言結いへの引き継ぎの図: docs/figs/handover_testbot02.svg（テストボット02 の部品のつながり）。実行: python tools/fig_handover_20261102.py"""
STYLE = "<style>.t{font:13px sans-serif;fill:#222} .ts{font:11px sans-serif;fill:#555} .th{font:bold 13px sans-serif;fill:#222} .arr{stroke:#666;stroke-width:1.4;fill:none}</style>"
DEFS = "<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#666'/></marker></defs>"
txt = lambda c, x, y, t: f"<text class='{c}' x='{x}' y='{y}' text-anchor='middle'>{t}</text>"
C = dict(blue=("#DCEBFA", "#3B7DD8"), coral=("#FCE3DA", "#D85A30"), green=("#DFF3DF", "#3C9A3C"), amber=("#FBEFD5", "#C98A12"), purple=("#ECE3F7", "#7E57C2"), teal=("#D8F1EE", "#1F8F84"), gray=("#EEEEEE", "#888"))


def box(x, y, w, h, col, lines):
    f, s = C[col]
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{f}' stroke='{s}'/>" + "".join(txt(c, x + w / 2, y + 17 + 15 * i, t) for i, (c, t) in enumerate(lines))


arr = lambda x1, y1, x2, y2: f"<path class='arr' d='M{x1} {y1} L{x2} {y2}' marker-end='url(#a)'/>"
b = txt("th", 480, 22, "テストボット02 の部品（Sin-CNN3 → 言結いへ引き継ぐもの）")
b += box(20, 40, 170, 50, "gray", [("th", "相手の発言"), ("ts", "+ 前の会話")])
b += box(230, 36, 200, 58, "purple", [("th", "読む"), ("ts", "状態遷移グレイン（16 状態、実験120）"), ("ts", "話題の切り替わり（実験121）")])
b += box(230, 104, 200, 58, "purple", [("th", "読む"), ("ts", "格逆算グレイン（実験119）"), ("ts", "課題の読み取り・知らない語")])
b += box(470, 36, 220, 126, "teal", [("th", "選ぶ（返事の候補）"), ("ts", "雑談 33 万発言から（実験115）"), ("ts", "重なり・次の発言の共鳴・口調"), ("ts", "遷移・前の話題・キャラ・時計"), ("ts", "体験談を下げる"), ("ts", "揺らした 10 通りの票 = 自信")])
b += box(730, 36, 210, 126, "green", [("th", "組み立てる"), ("ts", "共感（何に + 〜んですね）"), ("ts", "→ 内容 → 質問（遷移で決める）"), ("ts", "結びは基本丁寧（実験117）"), ("ts", "自信がない → どんな〜ですか？"), ("ts", "知らない語 → 聞き返す")])
b += box(230, 190, 200, 70, "amber", [("th", "キャラ"), ("ts", "プロフィール（持ち主が書く）"), ("ts", "人格グレイン（選び方の重み）")])
b += box(470, 190, 220, 70, "amber", [("th", "対話者グレイン"), ("ts", "名前・好き・予定・困りごと"), ("ts", "（保存は非公開の置き場だけ）")])
b += box(730, 190, 210, 70, "amber", [("th", "時計グレイン"), ("ts", "時間帯・月・季節（10 分刻み）"), ("ts", "表は持ち主が直せる")])
b += arr(190, 65, 228, 65) + arr(190, 70, 228, 130) + arr(430, 65, 468, 80) + arr(430, 133, 468, 120) + arr(690, 99, 728, 99)
b += arr(330, 190, 520, 164) + arr(580, 190, 580, 164) + arr(835, 190, 640, 164)
b += box(20, 290, 920, 78, "coral", [("th", "引き継ぐときに直すこと"), ("ts", "結びの付け替えの表は、実験だけに使う材料（Web 小説を含む）から作っている → ライセンスの分かる材料で作り直す"),
                                     ("ts", "返事の候補・遷移グレイン・格の好みは RealPersonaChat・JMultiWOZ（CC BY-SA 4.0）から → 表示と同じ条件で"),
                                     ("ts", "選ぶ部分は数え上げ（光の器ではない）→ 言結いのグレイン（光の部品）に置き換えていく")])
svg = f"<svg viewBox='0 0 960 385' xmlns='http://www.w3.org/2000/svg'>{STYLE}{DEFS}<rect width='960' height='385' fill='#fff'/>{b}</svg>"
open("docs/figs/handover_testbot02.svg", "w", encoding="utf-8").write(svg)
print("ok")
