"""会話の中で示した文章樹の図を SVG ファイルにする（持ち主の依頼 2026-11-02「画像類をまとめて GitHub にプッシュして」）。
出力: docs/figs/urashima_sentence1_tree.svg（1 文目の木）、docs/figs/story_tree_structure.svg（文章樹の構造）、docs/figs/story_structure_tree_v2.svg（構造木と答えまでの流れ）。
実行: python tools/fig_story_trees.py"""
import os

STYLE = """<style>
.c-coral rect{fill:#FCE3DA;stroke:#D85A30} .c-blue rect{fill:#DCEBFA;stroke:#3B7DD8} .c-amber rect{fill:#FBEFD5;stroke:#C98A12}
.c-purple rect{fill:#ECE3F7;stroke:#7E57C2} .c-teal rect{fill:#D8F1EE;stroke:#1F8F84} .c-gray rect{fill:#EEEEEE;stroke:#888}
.c-green rect{fill:#DFF3DF;stroke:#3C9A3C} rect{stroke-width:1.2}
.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}
.arr{stroke:#666;stroke-width:1.4} svg{background:#fff}
</style>"""
DEFS = "<defs><marker id='a' viewBox='0 0 10 10' refX='5' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#666'/></marker></defs>"


def box(cls, x, y, w, h, lines):
    """色付きの箱と、中の文字（1 行目は太字、2 行目からは小さめ）。"""
    s = f"<g class='{cls}'><rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8'/>"
    for i, (c, t) in enumerate(lines):
        s += f"<text class='{c}' x='{x + w / 2}' y='{y + 20 + 19 * i}' text-anchor='middle'>{t}</text>"
    return s + "</g>"


arr = lambda x1, y1, x2, y2, dash=False: f"<path class='arr' d='M{x1} {y1} L{x2} {y2}' fill='none' marker-end='url(#a)'{' stroke-dasharray=\"5 4\"' if dash else ''}/>"
txt = lambda c, x, y, t: f"<text class='{c}' x='{x}' y='{y}' text-anchor='middle'>{t}</text>"
svg = lambda w, h, body: f"<svg viewBox='0 0 {w} {h}' xmlns='http://www.w3.org/2000/svg'>{STYLE}{DEFS}<rect width='{w}' height='{h}' fill='#fff'/>{body}</svg>"
os.makedirs("docs/figs", exist_ok=True)
# 1. 1 文目の木（根 = 述語が上）
b = (box("c-coral", 230, 14, 260, 62, [("th", "根: 動詞句（述語）"), ("t", "暮らし て い まし た 。")])
     + txt("ts", 360, 92, "動詞 暮らす + 助詞 て + 動詞(補) いる + 助動詞 ます + 助動詞 た + 句点")
     + box("c-teal", 40, 150, 200, 62, [("th", "副詞句（いつ）"), ("t", "昔々 、")])
     + box("c-blue", 420, 150, 240, 62, [("th", "名詞句（主語・が）"), ("t", "漁師 が")])
     + box("c-purple", 300, 300, 220, 62, [("th", "連体修飾（名前）"), ("t", "浦島太郎 と いう")]) + txt("ts", 410, 380, "固有名詞 浦島 + 太郎 ／ 格助詞 と + 動詞 いう")
     + box("c-amber", 560, 300, 140, 62, [("th", "連体修飾"), ("t", "優しい")]) + txt("ts", 630, 380, "形容詞")
     + arr(140, 150, 300, 78) + arr(540, 150, 430, 78) + arr(410, 300, 510, 214) + arr(630, 300, 580, 214)
     + txt("ts", 200, 122, "いつ") + txt("ts", 505, 122, "誰が（主語）") + txt("ts", 440, 262, "どんな") + txt("ts", 630, 262, "どんな"))
open("docs/figs/urashima_sentence1_tree.svg", "w", encoding="utf-8").write(svg(720, 400, b))
# 2. 文章樹の構造（文章 → 文の根 → 文節、名前の名札、節グレインの持つもの）
b = (box("c-gray", 250, 10, 260, 44, [("th", "文章（浦島太郎 20 文）")])
     + box("c-coral", 40, 90, 180, 52, [("th", "1 文目の根（述語）"), ("ts", "暮らしていました")])
     + box("c-coral", 290, 90, 180, 52, [("th", "… 各文の根 …"), ("ts", "見つけます・渡します")])
     + box("c-coral", 540, 90, 180, 52, [("th", "20 文目の根"), ("ts", "なっていました")])
     + arr(330, 54, 150, 88) + arr(380, 54, 380, 88) + arr(430, 54, 610, 88)
     + box("c-blue", 20, 190, 100, 44, [("ts", "いつ"), ("t", "昔々")]) + box("c-blue", 135, 190, 100, 44, [("ts", "が（主語）"), ("t", "漁師")])
     + box("c-purple", 90, 280, 120, 44, [("ts", "という（名前）"), ("t", "浦島太郎")]) + box("c-amber", 220, 280, 80, 44, [("ts", "どんな"), ("t", "優しい")])
     + arr(70, 190, 110, 144) + arr(185, 190, 150, 144) + arr(150, 280, 175, 236) + arr(260, 280, 200, 236)
     + box("c-blue", 500, 190, 110, 44, [("ts", "は（主語）"), ("t", "太郎")]) + box("c-blue", 625, 190, 120, 44, [("ts", "に（なる先）"), ("t", "おじいさん")])
     + box("c-amber", 640, 280, 90, 44, [("ts", "どんな"), ("t", "真っ白")]) + box("c-amber", 500, 280, 110, 44, [("ts", "どんな"), ("t", "若かった")])
     + arr(555, 190, 610, 144) + arr(685, 190, 650, 144) + arr(685, 280, 685, 236) + arr(555, 280, 555, 236)
     + "<path class='arr' d='M150 326 C 300 420, 480 400, 540 236' fill='none' stroke-dasharray='5 4'/>" + txt("ts", 360, 405, "名前の名札：太郎 ⊂ 浦島太郎（文をまたぐ）")
     + txt("ts", 380, 440, "質問も同じ形の木にして、疑問の語 → 答えの節の役割（手で作った表 → 実験90 で学んだフィルター）")
     + box("c-teal", 40, 450, 680, 92, [("th", "節グレイン（文節 1 つごと）が持つもの"), ("t", "自分の語 ・ 役割（が・を・へ・という・どんな・引用） ・ 子の節"), ("t", "枝葉の語（深いほど薄く 0.7 倍） ・ 節の光（名詞の語板 v4 を明るさで足す）")]))
open("docs/figs/story_tree_structure.svg", "w", encoding="utf-8").write(svg(760, 560, b))
# 3. 構造木と答えまでの流れ
b = (txt("th", 200, 20, "文章樹（1 文章の語板）") + box("c-gray", 110, 32, 180, 40, [("th", "根: 文章（20 文）")])
     + box("c-coral", 10, 104, 120, 48, [("th", "幹: 1 文目"), ("ts", "述語 暮らす")]) + box("c-coral", 140, 104, 120, 48, [("th", "幹: …"), ("ts", "見つける・渡す")])
     + box("c-coral", 270, 104, 120, 48, [("th", "幹: 20 文目"), ("ts", "述語 成る")])
     + arr(160, 72, 80, 102) + arr(200, 72, 200, 102) + arr(240, 72, 320, 102)
     + box("c-blue", 10, 190, 70, 42, [("ts", "いつ"), ("t", "昔々")]) + box("c-blue", 90, 190, 70, 42, [("ts", "が（主語）"), ("t", "漁師")])
     + box("c-purple", 60, 262, 90, 42, [("ts", "という"), ("t", "浦島太郎")]) + box("c-amber", 160, 262, 60, 42, [("ts", "どんな"), ("t", "優しい")])
     + box("c-blue", 240, 190, 70, 42, [("ts", "は（主語）"), ("t", "太郎")]) + box("c-blue", 320, 190, 80, 42, [("ts", "に（成る先）"), ("t", "おじいさん")])
     + box("c-amber", 240, 262, 70, 42, [("ts", "どんな"), ("t", "若かった")]) + box("c-amber", 320, 262, 80, 42, [("ts", "どんな"), ("t", "真っ白な")])
     + arr(45, 190, 60, 154) + arr(125, 190, 90, 154) + arr(105, 262, 120, 234) + arr(190, 262, 140, 234)
     + arr(285, 190, 320, 154) + arr(355, 190, 340, 154) + arr(275, 262, 275, 234) + arr(360, 262, 360, 234)
     + "<path class='arr' d='M105 306 C 160 360, 250 350, 268 234' fill='none' stroke-dasharray='5 4'/>" + txt("ts", 190, 350, "名前の名札: 太郎 ⊂ 浦島太郎")
     + box("c-teal", 10, 380, 390, 100, [("th", "節グレイン（文節 1 つ）の光"), ("ts", "節の光 = 語の名札 1.0 + 役割 1.0 + 語板 v4 0.5"), ("ts", "枝葉の光 = 自分 + 下の枝（1 段ごとに 0.7 倍）"),
                                         ("ts", "まわりの光 = 親の枝葉 − 自分の枝葉 + 0.5 × (根の枝葉 − 自分の枝葉)")])
     + txt("th", 600, 20, "質問から答えへ") + box("c-gray", 470, 32, 260, 40, [("t", "質問「真っ白なおじいさんは誰？」")])
     + box("c-purple", 470, 96, 125, 58, [("th", "疑問フィルター"), ("ts", "誰 → が・は・に")]) + box("c-green", 605, 96, 125, 58, [("th", "バッファ"), ("ts", "手がかり（述語 2 倍）")])
     + arr(560, 72, 532, 94) + arr(640, 72, 667, 94)
     + box("c-coral", 470, 190, 260, 52, [("th", "1. 幹を選ぶ（枝葉の光の重なり）"), ("ts", "20 文目 0.32、ほか 0.06 以下")]) + arr(667, 156, 640, 188)
     + box("c-blue", 470, 272, 260, 70, [("th", "3. 介入: 節ごとに"), ("ts", "まわりの光の説明 × 役割の合い"), ("ts", "（役割フィルター・否定の向き）")])
     + arr(600, 244, 600, 270) + arr(532, 156, 520, 270)
     + box("c-amber", 520, 372, 160, 44, [("th", "答え: 太郎")]) + arr(600, 344, 600, 370)
     + "<path class='arr' d='M470 300 C 430 300, 420 230, 402 220' fill='none' stroke-dasharray='5 4' marker-end='url(#a)'/>"
     + txt("ts", 600, 450, "手の所: 文の木（GiNZA）・役割の対応表・名札") + txt("ts", 600, 470, "学んだ所: 疑問フィルター・役割フィルター")
     + txt("ts", 390, 520, "浦島太郎 30 問: 手の決まり + 述語 2 倍 29 ／ 学んだフィルター + 述語 2 倍 26"))
open("docs/figs/story_structure_tree_v2.svg", "w", encoding="utf-8").write(svg(780, 540, b))
print("ok")
