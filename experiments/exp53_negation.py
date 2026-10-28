"""実験53: 「〜ではない」の表し方（実験52 の続き、持ち主との相談 2026-10-28）

着想: 持ち主「〇〇ではない、を表す方法はいくつか考えてるんだけど、結構難しい。問題は範囲の広さなので、0 − 特徴や属性でマイナス差分がいいのかなぁ」。
      「赤くない」は青・緑・紫…を全部含む（範囲が広い）ので、1 つの値を光らせる形では表せない。
方法: 実験52 と同じ 7 物のグレイン（3 seed）。否定の手がかり（赤くない・甘くない・果物ではない・加熱しない・苦くない）を含む文 5 つ。
      否定の手がかり (観点 v, 値 x) の扱い方 5 通り（肯定の手がかりはすべて掛け算、重み 1）:
        無視           : 否定の手がかりを使わない
        読み落とす     : 否定を見落として、肯定（赤い）として掛ける（比べ: 否定を理解しないとどうなるか）
        割り算         : 割合 p で割る（式のマイナスのべき p^{−1}）。p が 0 に近い物で値が爆発する
        0 − 特徴       : 持ち主の案。足し算の点数で、否定の手がかりの割合を引く（点数 = Σ 肯定の p − Σ 否定の p、0 未満は 0）
        1 − p（補い）  : 否定の値に行かなかった残りの光の割合を掛ける（ビームスプリッタのもう片方の出口のような形）
指標: 正しい物が 1 番か、正しい物の割合。結果の置き場: results/exp53.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp53_negation.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grain import Grain

src = open(os.path.join(os.path.dirname(__file__), "exp51_new_grain.py"), encoding="utf-8").read()
ns = {"__file__": os.path.join(os.path.dirname(__file__), "exp51_new_grain.py")}; exec(src.split("\nout = ")[0], ns)
VIEWS, VAL, VN, TABLE, target = ns["VIEWS"], ns["VAL"], ns["VN"], ns["TABLE"], ns["target"]
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
OBJS = list(TABLE)
POS = {"丸い": ("形", "丸い"), "甘い": ("味", "甘い"), "かじった": ("食べ方", "生"), "細長い": ("形", "長い"), "焼いた": ("食べ方", "加熱"), "赤い": ("色", "赤"),
       "緑の": ("色", "緑"), "サラダ": ("食べ方", "生")}
NEG = {"赤くない": ("色", "赤"), "甘くない": ("味", "甘い"), "果物ではない": ("分類", "果物"), "加熱しない": ("食べ方", "加熱"), "苦くない": ("味", "苦い")}
SENT = [("赤くない 丸い 甘い ○○ を かじった", "リンゴ"),          # 青リンゴ・黄色いリンゴ（イチゴ・トマトは赤ばかり）
        ("甘くない 細長い ○○ を 焼いた", "ナス"),
        ("果物ではない 赤い ○○", "トマト"),
        ("加熱しない 緑の ○○ を サラダ に", "キュウリ"),
        ("苦くない 緑の 細長い ○○ を 焼いた", "キュウリ")]
out = "results/exp53.txt"
for seed in range(3):
    G = {o: Grain(len(VN), len(VAL), seed=seed + 10 * k) for k, o in enumerate(OBJS)}
    for o in OBJS:
        G[o].fit(target(o), steps=600)
    A = {o: G[o].answer() for o in OBJS}
    frac = lambda o, v, x: float(A[o][VN.index(v), VAL.index(x)] / A[o][VN.index(v), [VAL.index(y) for y in VIEWS[v]]].sum())
    rows = []
    for text, ans in SENT:
        tok = text.split(); pos = [POS[t] for t in tok if t in POS]; neg = [NEG[t] for t in tok if t in NEG]
        lp = {o: sum(math.log(max(frac(o, v, x), 1e-4)) for v, x in pos) for o in OBJS}
        sc = {"無視": {o: math.exp(lp[o]) for o in OBJS},
              "読み落とす": {o: math.exp(lp[o] + sum(math.log(max(frac(o, v, x), 1e-4)) for v, x in neg)) for o in OBJS},
              "割り算": {o: math.exp(lp[o] - sum(math.log(max(frac(o, v, x), 1e-4)) for v, x in neg)) for o in OBJS},
              "0 − 特徴": {o: max(0.0, sum(frac(o, v, x) for v, x in pos) - sum(frac(o, v, x) for v, x in neg)) + 1e-9 for o in OBJS},
              "1 − p": {o: math.exp(lp[o] + sum(math.log(max(1 - frac(o, v, x), 1e-4)) for v, x in neg)) for o in OBJS}}
        row = dict(text=text, answer=ans)
        for k, s in sc.items():
            z = sum(s.values()); p = {o: x / z for o, x in s.items()}; r = sorted(p, key=p.get, reverse=True)
            row[k] = dict(top=r[0], ok=r[0] == ans, p_answer=p[ans], p_top=p[r[0]])
        rows.append(row)
        print(seed, text, "→", " / ".join(f"{k}: {row[k]['top']} {100 * row[k]['p_top']:.0f}%{'○' if row[k]['ok'] else '×'}" for k in sc), flush=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(exp="53", seed=seed, rows=rows), ensure_ascii=False) + "\n")
