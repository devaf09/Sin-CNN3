"""実験52: 周りの語の属性を掛け合わせて、空いた所（○○）の語を当てる（持ち主の案、2026-10-28）

着想: 持ち主「文章中のあるトークンは周囲のトークンと属性でつながる。逆説的にその語は周囲の語から属性でつながって、そこに無い物になる。
      関係性の薄い属性は掛け算で消えていくから、本質的に関連する物が残る。助詞は次の語との関連度を強める（n × 1.2 など）。
      例: 食べ物 ○、果物 △（ちょっとだけ）、料理が必要 ○ → 一番近いのはナス？」
      学び 21 の「手がかりの掛け算」（逆引きの明るさを掛ける = 素朴ベイズ）を、文の中の位置と重みつきに広げたもの。
方法: 実験51 のグレイン（1 物 = 1 枚の器）を 7 物（リンゴ・トマト・イチゴ・キュウリ・ナス・バナナ・ピーマン）ぶん学習。
      周りの語は「観点 → 値」の手がかりを持つ（手で作った小さな表 CLUE。将来は語ごとのグレインから出す）。手がかりのない語（昨日・とても・助詞）は何も出さない。
      候補 c の点数 = Π_手がかり（c のグレインがその観点でその値に送る明るさの割合）^重み。重み = 3 通り:
        重み 1    : すべて 1（学び 21 と同じ）
        距離の重み: ○○ のすぐ隣（助詞をはさんで 1 語先まで）は 1.2（持ち主の案）、そこから 1 語離れるごとに × 0.7
        足し算    : 比べ。掛け算の代わりに割合を足す
      光一発で行うなら: 手がかりごとの明るさの割合を、光を順に通す吸収板の透過率にすると、通った光 = 透過率の掛け算。
      重み（べき）は吸収板の厚さ（透過率 T の板を w 倍の厚さにすると T^w、ベール・ランベルトの法則）。
逆回し: 当てた物のグレインに観点の波を入れると、その物の属性が出る（属性 → 語 → 属性）。
指標: 正しい物が 1 番か、1 番の割合（確信度）、2 番との差。3 seed。結果の置き場: results/exp52.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp52_cloze_product.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grain import Grain

src = open(os.path.join(os.path.dirname(__file__), "exp51_new_grain.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(os.path.dirname(__file__), "exp51_new_grain.py")}; exec(src.split("\nout = ")[0], ns)                                       # VIEWS・TABLE・target を実験51 から借りる
VIEWS, VAL, VN, TABLE, target = ns["VIEWS"], ns["VAL"], ns["VN"], ns["TABLE"], ns["target"]
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
OBJS = list(TABLE)
CLUE = {"蒸し器": [("食べ方", "加熱")], "煮た": [("食べ方", "加熱")], "焼いた": [("食べ方", "加熱")], "サラダ": [("食べ方", "生")], "かじった": [("食べ方", "生")],
        "デザート": [("味", "甘い"), ("分類", "果物")], "おかず": [("分類", "野菜")], "紫の": [("色", "紫")], "赤い": [("色", "赤")], "緑の": [("色", "緑")],
        "黄色い": [("色", "黄")], "細長い": [("形", "長い")], "丸い": [("形", "丸い")], "甘い": [("味", "甘い")], "酸っぱい": [("味", "酸っぱい")], "苦い": [("味", "苦い")],
        "夕ご飯": [("分類", "野菜")], "果物": [("分類", "果物")]}
SENT = [("夕ご飯 は ○○ を 蒸し器 で 蒸した", "ナス"),                                   # 持ち主の例に近い形
        ("紫の ○○ を 焼いた", "ナス"),
        ("デザート に ○○ を かじった 。 赤い", "リンゴ"),
        ("サラダ に 緑の 細長い ○○ を 入れた", "キュウリ"),
        ("赤い 酸っぱい ○○ を 煮た", "トマト"),
        ("黄色い 細長い 甘い ○○", "バナナ"),
        ("苦い ○○ を おかず に 焼いた", "ピーマン"),
        ("昨日 とても 赤い 丸い ○○ を デザート に した", "イチゴ"),
        ("赤い ○○ を 蒸し器 で 煮た 。 昨日 の デザート は 甘い 果物 だった", "トマト")]   # 遠くに別の話（デザート・甘い・果物）がある


def weights(tokens, kind):
    b = tokens.index("○○"); w = []
    for i, t in enumerate(tokens):
        if t not in CLUE:
            continue
        d = abs(i - b)
        if kind == "距離の重み":
            w.append((t, 1.2 if d <= 2 else 1.2 * 0.7 ** (d - 2)))
        else:
            w.append((t, 1.0))
    return w


out = "results/exp52.txt"
for seed in range(3):
    G = {o: Grain(len(VN), len(VAL), seed=seed + 10 * k) for k, o in enumerate(OBJS)}
    for o in OBJS:
        G[o].fit(target(o), steps=600)
    A = {o: G[o].answer() for o in OBJS}
    frac = lambda o, v, x: float(A[o][VN.index(v), VAL.index(x)] / A[o][VN.index(v), [VAL.index(y) for y in VIEWS[v]]].sum())
    rows = []
    for text, ans in SENT:
        tok = text.split(); row = dict(text=text, answer=ans)
        for kind in ["重み 1", "距離の重み", "足し算"]:
            ws = weights(tok, "重み 1" if kind == "足し算" else kind)
            if kind == "足し算":
                sc = {o: sum(frac(o, v, x) for t, w in ws for v, x in CLUE[t]) for o in OBJS}
            else:
                sc = {o: math.exp(sum(w * math.log(max(frac(o, v, x), 1e-4)) for t, w in ws for v, x in CLUE[t])) for o in OBJS}
            z = sum(sc.values()); p = {o: s / z for o, s in sc.items()}
            rank = sorted(p, key=p.get, reverse=True)
            row[kind] = dict(top=rank[0], ok=rank[0] == ans, p_top=p[rank[0]], p_answer=p[ans], margin=p[rank[0]] - p[rank[1]], second=rank[1])
        top = row["距離の重み"]["top"]                                               # 逆回し: 当てた物の属性
        row["逆回し"] = {v: VIEWS[v][int(A[top][i, [VAL.index(y) for y in VIEWS[v]]].argmax())] for i, v in enumerate(VN)}
        rows.append(row)
        print(seed, text, "→", " / ".join(f"{k}: {row[k]['top']}（{100 * row[k]['p_top']:.0f}%, 2 番 {row[k]['second']}）{'○' if row[k]['ok'] else '×'}" for k in ["重み 1", "距離の重み", "足し算"]),
              "| 逆回し", row["逆回し"], flush=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(exp="52", seed=seed, rows=rows), ensure_ascii=False) + "\n")
