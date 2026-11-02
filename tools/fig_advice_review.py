"""別の AI の助言（2026-11-02）の 5 つの提案を図にする。docs/advice_review_20261102.md から参照。
出力: docs/figs/advice_1_resonance.svg、advice_2_pipeline.svg、advice_3_refractory.svg、advice_4_binding.svg、advice_5_reference.svg
実行: python tools/fig_advice_review.py"""
import math

STYLE = """<style>.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}
.arr{stroke:#666;stroke-width:1.4;fill:none} .ax{stroke:#999;stroke-width:1}</style>"""
DEFS = ("<defs><marker id='a' viewBox='0 0 10 10' refX='5' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#666'/></marker>"
        "<marker id='r' viewBox='0 0 10 10' refX='5' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#D85A30'/></marker></defs>")
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
C = dict(blue=("#DCEBFA", "#3B7DD8"), coral=("#FCE3DA", "#D85A30"), green=("#DFF3DF", "#3C9A3C"), amber=("#FBEFD5", "#C98A12"), purple=("#ECE3F7", "#7E57C2"), gray=("#EEEEEE", "#888"), teal=("#D8F1EE", "#1F8F84"))


def box(x, y, w, h, col, lines):
    f, s = C[col]
    return f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8' fill='{f}' stroke='{s}'/>" + "".join(txt(c, x + w / 2, y + 19 + 17 * i, t) for i, (c, t) in enumerate(lines))


def wave(x, y, w, amp, k, col, ph=0):
    pts = " ".join(f"{x + i:.1f},{y + amp * math.sin(2 * math.pi * k * i / w + ph):.1f}" for i in range(0, int(w) + 1, 2))
    return f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='1.6'/>"


def save(name, w, h, body):
    open(f"docs/figs/{name}.svg", "w", encoding="utf-8").write(f"<svg viewBox='0 0 {w} {h}' xmlns='http://www.w3.org/2000/svg'>{STYLE}{DEFS}<rect width='{w}' height='{h}' fill='#fff'/>{body}</svg>")


# 1. 共鳴パーシング: 助詞の「要求の波」と動詞の「受け入れの波」が合う所に係り受けを張る
b = txt("th", 450, 24, "提案 1: 共鳴パーシング（係り受けを波の重なりで決める）")
words = [("太郎", "が", 60), ("浜辺", "で", 250), ("亀", "を", 440), ("見つける", "", 690)]
for w, p, x in words[:3]:                                                                       # 係り受けの矢印は上に弧で
    b += f"<path class='arr' d='M{x + 75} 108 C {x + 120} 50, 700 50, 755 106' stroke='#3C9A3C' stroke-width='2' marker-end='url(#a)'/>"
for w, p, x in words:
    b += box(x, 110, 150, 44, "blue" if p else "coral", [("th", w + p)])
for w, p, x in words[:3]:
    k = {"が": 3, "で": 5, "を": 4}[p]
    b += wave(x + 10, 180, 130, 10, k, "#3B7DD8") + txt("ts", x + 75, 210, f"要求の波（{p}）")
b += wave(700, 175, 130, 8, 3, "#D85A30") + wave(700, 190, 130, 8, 4, "#D85A30") + wave(700, 205, 130, 8, 5, "#D85A30")
b += txt("ts", 765, 228, "受け入れの波（が・を・で）")
b += txt("ts", 450, 252, "周波数が合う組（が ↔ 見つける の が 受け など）で重なりが強くなる → そこに係り受けの矢印（緑）を張る")
b += box(60, 270, 780, 70, "gray", [("th", "材料はもうある"), ("ts", "実験91: 助詞が前後に求める品詞（差分）  ／  実験93: 同じ助詞を働きごとに分ける（が 0.98）"),
                                     ("ts", "語板 v5 の動詞の観点 V:が・V:を・V:に（その動詞がどの格にどんな名詞をとるか）")])
save("advice_1_resonance", 900, 360, b)
# 2. 3 段の生成
b = txt("th", 450, 24, "提案 2: 命題 → 骨格（格フレーム）→ 並列の文節展開")
b += box(20, 50, 230, 110, "purple", [("th", "1. 命題を選ぶ"), ("ts", "バッファから述語を 1 つ"), ("ts", "例: 見つける"), ("ts", "（何を言いたいか）"), ("ts", "まだ無い")])
b += box(330, 50, 250, 110, "amber", [("th", "2. 骨格を作る"), ("ts", "述語の要求する格のスロット"), ("ts", "[が] [で] [を] + 見つける"), ("ts", "+ つなぎ方・文の結び（実験105）"), ("ts", "→ 実験106 で試す")])
b += box(660, 50, 220, 110, "green", [("th", "3. 並列に展開"), ("ts", "スロットごとに局所で"), ("ts", "太郎は / 浜辺で / 亀を"), ("ts", "/ 見つけます。"), ("ts", "実験105: 60〜80%")])
b += f"<path class='arr' d='M250 105 L328 105' marker-end='url(#a)'/><path class='arr' d='M580 105 L658 105' marker-end='url(#a)'/>"
b += txt("ts", 455, 190, "スロットの中身を埋める名詞は、語板 v5 の V:が・V:を…（その格に来やすい名詞の仲間）から選べる")
b += txt("ts", 455, 210, "注意: は／が、つなぎ方、文の結び（丁寧さ・人称）は述語からは決まらない → 話の流れと話者関係から")
save("advice_2_pipeline", 900, 230, b)
# 3. 不応期
b = txt("th", 450, 24, "提案 3: 不応期（使った語・品詞を一時的に抑えて、輪を切る）")
x0, y0 = 60, 60
b += txt("th", 230, 50, "抑えなし: て → 動詞 → て → … と回り続ける") + txt("th", 680, 50, "抑えあり: 使った所に逆の位相の光を足す")
seq1 = ["名詞", "て", "動詞", "て", "動詞", "て", "動詞"]; seq2 = ["名詞", "て", "動詞", "ます", "。"]
for i, s in enumerate(seq1):
    b += box(20 + i * 62, 70, 54, 32, "coral" if s in ("て", "動詞") else "blue", [("ts", s)])
for i, s in enumerate(seq2):
    b += box(480 + i * 80, 70, 70, 32, "green" if s in ("ます", "。") else "blue", [("ts", s)])
# 抑えの強さ（時間で戻る）
b += f"<line class='ax' x1='480' y1='200' x2='880' y2='200'/>" + txt("ts", 480, 122, "て の出やすさ（高いほど出やすい）", "start")
pts = []
for i in range(0, 401, 4):
    t = i / 80; v = 1.0 if t < 1 else 1 - 0.9 * math.exp(-(t - 1) * 0.9)
    pts.append(f"{480 + i},{200 - 70 * v:.1f}")
b += f"<polyline points='{' '.join(pts)}' fill='none' stroke='#D85A30' stroke-width='2'/>" + txt("ts", 560, 222, "て を使った直後は低く、時間で戻る（× 0.9 ずつ回復）")
b += txt("ts", 230, 140, "実験82 の正引きの連鎖で起きた輪") + txt("ts", 230, 160, "局所の相性だけだと、文の終わりへ向かえない")
save("advice_3_refractory", 900, 240, b)
# 4. 束縛
b = txt("th", 450, 24, "提案 4: 名札 ⊗ 役割（同一性）と 属性の光（意味）を分けて持つ")
b += box(30, 50, 250, 70, "blue", [("th", "名札 ⊗ 役割"), ("ts", "太郎 ⊙ 主語（位相の掛け算）"), ("ts", "誰が・何を を誤りなく照合")])
b += txt("th", 315, 92, "+")
b += box(350, 50, 250, 70, "teal", [("th", "属性の光"), ("ts", "語板の光（人・男・漁師らしさ）"), ("ts", "言い換え・どんな人か に使う")])
b += txt("th", 635, 92, "=")
b += box(670, 50, 200, 70, "amber", [("th", "節の表現"), ("ts", "照合は名札で"), ("ts", "連想は属性で")])
b += box(30, 150, 840, 60, "gray", [("th", "今の状況: 実験103 で名札 ⊙ 役割 は済み（取り出し 98%、誤爆は問題でなかった）"),
                                    ("ts", "外れたのは「質問がどの枠か」を選ぶ所。属性の光を別の層で足すのは未試験 → 枠の選び方に使えるかもしれない")])
save("advice_4_binding", 900, 230, b)
# 5. 大域の参照光
b = txt("th", 450, 24, "提案 5: 大域の参照光（今何を目指すかを、全部の節に配る）")
b += box(350, 45, 200, 50, "purple", [("th", "意図の光"), ("ts", "述語・問いの中心")])
for i, w in enumerate(["太郎は", "浜辺で", "亀を", "見つけます。"]):
    x = 60 + i * 200
    b += box(x, 160, 150, 40, "blue", [("t", w)]) + f"<path class='arr' d='M450 97 L{x + 75} 158' stroke='#7E57C2' stroke-dasharray='5 4' marker-end='url(#a)'/>"
b += box(60, 225, 780, 60, "gray", [("th", "これまでの結果: 効くのは「何を目指すか」の情報そのもの"),
                                    ("ts", "実験97: 質問の述語を 2 倍 → 29/30（効いた） ／ 実験105: 係る先の述語を全スロットに配る → +2〜3 点だけ")])
save("advice_5_reference", 900, 300, b)
print("ok")
