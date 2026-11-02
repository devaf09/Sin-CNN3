"""実験103（ホログラフィックな物語の光）の図: docs/figs/holographic_story_103.svg
上: 結びつけ（掛け算）・重ね合わせ（足し算）・時刻の鍵・取り出し（逆の位相）の流れ。下: 結果（光の長さ D ごと）。
実行: python tools/fig_holographic_103.py（results/exp103.txt の最後の行を読む）"""
import json

STYLE = """<style>
.c-coral rect{fill:#FCE3DA;stroke:#D85A30} .c-blue rect{fill:#DCEBFA;stroke:#3B7DD8} .c-amber rect{fill:#FBEFD5;stroke:#C98A12}
.c-purple rect{fill:#ECE3F7;stroke:#7E57C2} .c-teal rect{fill:#D8F1EE;stroke:#1F8F84} .c-gray rect{fill:#EEEEEE;stroke:#888}
.c-green rect{fill:#DFF3DF;stroke:#3C9A3C} rect{stroke-width:1.2}
.t{font:14px sans-serif;fill:#222} .ts{font:12px sans-serif;fill:#555} .th{font:bold 14px sans-serif;fill:#222}
.arr{stroke:#666;stroke-width:1.4} .ax{stroke:#999;stroke-width:1}
</style>"""
DEFS = "<defs><marker id='a' viewBox='0 0 10 10' refX='5' refY='5' markerWidth='6' markerHeight='6' orient='auto'><path d='M0,0 L10,5 L0,10 z' fill='#666'/></marker></defs>"


def box(cls, x, y, w, h, lines):
    s = f"<g class='{cls}'><rect x='{x}' y='{y}' width='{w}' height='{h}' rx='8'/>"
    for i, (c, t) in enumerate(lines):
        s += f"<text class='{c}' x='{x + w / 2}' y='{y + 20 + 18 * i}' text-anchor='middle'>{t}</text>"
    return s + "</g>"


arr = lambda x1, y1, x2, y2: f"<path class='arr' d='M{x1} {y1} L{x2} {y2}' fill='none' marker-end='url(#a)'/>"
txt = lambda c, x, y, t, a="middle": f"<text class='{c}' x='{x}' y='{y}' text-anchor='{a}'>{t}</text>"
runs = json.loads(open("results/exp103.txt", encoding="utf-8").read().strip().splitlines()[-1])["runs"]
# ---- 上: 仕組み ----
b = txt("th", 400, 22, "構造と時の順を 1 枚の光に畳む（結びつけ = 掛け算、重ね合わせ = 足し算）")
b += box("c-blue", 20, 40, 170, 80, [("th", "結びつけ ⊙"), ("ts", "役割の光 × 語の光"), ("ts", "（位相の板を通す）"), ("ts", "主語 ⊙ 太郎")])
b += box("c-amber", 215, 40, 170, 80, [("th", "枠（出来事）＋"), ("ts", "述語 ⊙ 見つける"), ("ts", "+ 主語 ⊙ 太郎"), ("ts", "+ 目的語 ⊙ 亀 + 袋")])
b += box("c-coral", 410, 40, 170, 80, [("th", "時刻の鍵 τ^k"), ("ts", "1 文ごとに位相を回す"), ("ts", "文の中は σ^j"), ("ts", "次 = 1 歩回すだけ")])
b += box("c-purple", 605, 40, 175, 80, [("th", "物語の光（1 枚）"), ("ts", "Σ τ^k σ^j ⊙ 枠"), ("ts", "80 枠・255 語が"), ("ts", "長さ D の光に重なる")])
b += arr(190, 80, 213, 80) + arr(385, 80, 408, 80) + arr(580, 80, 603, 80)
b += box("c-green", 20, 150, 370, 62, [("th", "取り出し（逆の位相を掛ける）"), ("ts", "物語 ⊙ τ^-k σ^-j ⊙ 主語^-1 → 太郎 + ばらばらの雑音"), ("ts", "→ 語の名札の一覧と重ねて、いちばん近い語（掃除）")])
b += box("c-teal", 410, 150, 370, 62, [("th", "質問"), ("ts", "手がかり（主語 ⊙ 太郎 + 述語 ⊙ 買い取る ×2 + 袋）"), ("ts", "→ どの枠（時刻）と重なるか → 答えの役割で取り出す")])
b += arr(690, 122, 600, 148) + arr(690, 122, 210, 148)
# ---- 下: 結果 ----
y0, H = 270, 150
b += txt("th", 400, y0 - 16, "結果（浦島太郎、述語ごとの枠 80 個）")
fr = [r for r in runs if r["mode"] == "枠"]
cols = [("取り出し正答（時刻・役割が分かるとき）", lambda r: r["recall"], "#3B7DD8"), ("次の出来事（19 問）", lambda r: r["next_ok"] / r["next_n"], "#1F8F84"),
        ("質問 30 問", lambda r: (r["q20"] + r["q10"]) / 30, "#D85A30")]
b += f"<line class='ax' x1='70' y1='{y0 + H}' x2='560' y2='{y0 + H}'/><line class='ax' x1='70' y1='{y0}' x2='70' y2='{y0 + H}'/>"
for v in (0, 0.5, 1):
    b += txt("ts", 62, y0 + H - v * H + 4, f"{int(v * 100)}%", "end") + f"<line x1='70' x2='560' y1='{y0 + H - v * H}' y2='{y0 + H - v * H}' stroke='#eee'/>"
for i, r in enumerate(fr):
    gx = 90 + i * 118
    for j, (name, f, c) in enumerate(cols):
        v = f(r); x = gx + j * 30
        b += f"<rect x='{x}' y='{y0 + H - v * H}' width='26' height='{v * H}' fill='{c}' rx='2'/>" + txt("ts", x + 13, y0 + H - v * H - 4, f"{v * 100:.0f}")
    b += txt("ts", gx + 42, y0 + H + 18, f"D = {r['D']}")
# 実験97（文章樹）の線
b += f"<line x1='70' x2='560' y1='{y0 + H - 29 / 30 * H}' y2='{y0 + H - 29 / 30 * H}' stroke='#D85A30' stroke-dasharray='5 4'/>" + txt("ts", 562, y0 + H - 29 / 30 * H + 4, "文章樹 29/30", "start")
for j, (name, _, c) in enumerate(cols):
    b += f"<rect x='590' y='{y0 + 40 + 24 * j}' width='14' height='14' fill='{c}'/>" + txt("ts", 610, y0 + 52 + 24 * j, name, "start")
b += txt("ts", 590, y0 + 130, "覚えること・順番は D を大きくすれば正しく出る。", "start")
b += txt("ts", 590, y0 + 148, "外れるのは「質問がどの枠か」を選ぶところ。", "start")
svg = f"<svg viewBox='0 0 960 460' xmlns='http://www.w3.org/2000/svg'>{STYLE}{DEFS}<rect width='960' height='460' fill='#fff'/>{b}</svg>"
open("docs/figs/holographic_story_103.svg", "w", encoding="utf-8").write(svg)
print("ok")
