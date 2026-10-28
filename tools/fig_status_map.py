"""図: 現状のマップ（基礎 → 記憶 → 推し量り → 組み合わせ → 文脈 → 会話）。部品ごとに、できた / 一部 / まだ を色で示す。
実行: python tools/fig_status_map.py → docs/images/status_map.png（中身を変えたら、下の COLS を書き換える）"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
OK, PART, TODO = "#b7e1b0", "#ffe39a", "#e6e6e6"
COLS = [
    ("1. 波の基礎", [("FFT の約束・漏れ・直交", OK, "01〜07"), ("1 本の周波数に振幅と位相", OK, "02・06"), ("回折・角スペクトル", OK, "09・12"), ("rfft・精度・標本化", TODO, "M0 の残り")]),
    ("2. 記憶", [("重ね書きの容量（式どおり）", OK, "08"), ("関係 A→B,C,D・ずらし", OK, "15・16・18"), ("単語ごとの格子（1 KB）", OK, "31"), ("別々に呼べる記憶 + 非線形\n容量が p ごとに N² 倍", OK, "47"), ("融合（明るさで足す）", OK, "34・36・37"), ("差分（明るさで引く）", OK, "44"), ("規模（名詞 80 × 観点 9）", PART, "28: 画素の板が要る")]),
    ("3. 推し量り", [("観点周波数・逆引き", OK, "21"), ("手がかりの掛け算", OK, "21・24・25"), ("2 つの意味（同音異義）", OK, "26"), ("知らない意味に気づく", OK, "27"), ("道具を関数として使う", OK, "32"), ("f2 のモード（AI の状態）", OK, "41"), ("観点の波は 2 ビットで足りる", OK, "42・43"), ("知らない物で新しいグレイン\n（合う率 < しきい値、直交）", OK, "51")]),
    ("4. 組み合わせ", [("どちらか片方だけ（カー効果）", OK, "14"), ("明るさを測って出し直す 2 段", OK, "38・40"), ("関係を何段たどっても濁らない\n（p 乗 + 利得のループ）", OK, "17・50"), ("カー効果で語順（強さをそろえ、
焦点面でかける）", PART, "54: 2-gram 越え"), ("両方そろったときだけ（完全に）", PART, "40: 42〜49%")]),
    ("5. 文脈", [("光の注意機構（部品）", OK, "49"), ("長い文脈（番地に並べる）", OK, "49: 4096 で 100%"), ("次の文字: 2-gram 越え", PART, "38: 27.4%"), ("3-gram 越え", TODO, "33.9% に届かず"), ("語の順番（位置）の入れ方", TODO, ""), ("トランスフォーマーより良い\n注意機構", TODO, "持ち主の方針")]),
    ("6. 会話", [("会話データ（非公開）", OK, "実会話 1.4 万・小説 10 万"), ("周りの語の属性の掛け算で\n空いた語を当てる", PART, "52: 手がかりは手作り"), ("トークンの分け方（形態素）", OK, "Sudachi で 1600 万語"), ("語板（会話から 400 語）", OK, "55: 表にほぼ追いつく"), ("動詞の語板・名詞→動詞→名詞", OK, "56"), ("語板が平らにした表に追いつく", OK, "58: 差 0.01〜0.03"), ("語板を数万語に（GPU で数分）", OK, "59: 名詞 2 万・動詞 3000"), ("名前を第三者 No.nnn に", TODO, ""), ("文を作る（生成）", TODO, ""), ("対話の評価", TODO, "")]),
]
fig, ax = plt.subplots(figsize=(19, 9), dpi=120); ax.set_xlim(0, 19); ax.set_ylim(-0.4, 9.4); ax.axis("off")
W, H, GAP = 2.9, 0.84, 0.1
for c, (title, items) in enumerate(COLS):
    x = 0.25 + c * 3.12
    ax.text(x + W / 2, 9.0, title, ha="center", va="center", fontsize=12, fontweight="bold")
    for r, (name, col, exp) in enumerate(items):
        y = 8.3 - r * (H + GAP) - H
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08", fc=col, ec="0.5", lw=0.8))
        ax.text(x + 0.1, y + H - 0.12, name, ha="left", va="top", fontsize=8.6)
        if exp:
            ax.text(x + W - 0.08, y + 0.08, ("実験 " if exp[0].isdigit() else "") + exp, ha="right", va="bottom", fontsize=7.2, color="0.3")
    if c < len(COLS) - 1:
        ax.annotate("", (x + W + 0.2, 9.0), (x + W - 0.1, 9.0), arrowprops=dict(arrowstyle="->", lw=1.5))
for i, (col, lab) in enumerate([(OK, "できた（式・実験で確かめた）"), (PART, "一部できた"), (TODO, "まだ・失敗")]):
    ax.add_patch(FancyBboxPatch((0.3 + i * 3.6, -0.3), 0.4, 0.3, boxstyle="round,pad=0.01", fc=col, ec="0.5"))
    ax.text(0.8 + i * 3.6, -0.15, lab, va="center", fontsize=9)
ax.text(18.7, -0.15, "Sin-CNN3 のいま（2026-10-28、実験 01〜60）", ha="right", va="center", fontsize=9, color="0.3")
fig.tight_layout(); fig.savefig("docs/images/status_map.png")
