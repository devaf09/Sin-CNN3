"""語板をでたらめに 100 個選び、それぞれが持つ属性の一覧を出す（持ち主の依頼 2026-10-31）。
語板 v3（名詞 3 万語・観点 9 × 値 45）。属性 = (観点, 値)。「持つ」= 観点の中での割合が 0.02 以上（値 45 個に平らに分けたとき 0.022 なので、それを超える所）。
観点ごとに割合の大きい順。多いときは上位 8 つを出し、残りは「以下〇個」。観点ごとの「その他」（上位 45 値の外）の割合も添える。
品詞（実験78 の語板）が分かる語には、品詞の割合も添える。出力: docs/board_attributes_100.md。実行: python tools/board_attributes.py"""
import os, random, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4)
PRIV = "D:/AI_git/Sin-CNN3_private"; TH = 0.02; SHOW = 8
Ps = []; WORDS = []
for k in (0, 1):
    M = torch.load(f"{PRIV}/models/goban_v3_part{k}_scratch.pt", weights_only=False); Ps.append(M["P"]); WORDS += M["words"]
VIEWS = M["views"]; INV = {(j, r): x for j, v in enumerate(VIEWS) for x, r in M["pos"][v].items()}
pick = sorted(random.Random(1031).sample(range(len(WORDS)), 100))                            # でたらめに 100 語（よく出る順の位置で並べる）
with torch.no_grad():
    B3 = run(torch.cat(Ps)[pick], 9, 54)
B3 = B3 / B3.sum(-1, keepdim=True)                                                          # 観点ごとの割合（その他を含めて 1）
POSM = torch.load(f"{PRIV}/models/goban_pos.pt", weights_only=False); PWI = {w: i for i, w in enumerate(POSM["words"])}
with torch.no_grad():
    PB = run(POSM["P"][[PWI[WORDS[i]] for i in pick if WORDS[i] in PWI]], len(POSM["views"]), POSM["nout"])[:, 0, POSM["phys"][0][:14]]   # 観点 1（大分類）の出口（ずらしてある）
PB = PB / PB.sum(-1, keepdim=True); pk = [i for i in pick if WORDS[i] in PWI]
L = ["# 語板 100 個の属性の一覧（語板 v3、でたらめに選んだ名詞）", "",
     f"持ち主の依頼（2026-10-31）。作り方: tools/board_attributes.py。属性 = 観点 + 値。観点の中での割合が {TH} 以上の値を「持つ」とし、大きい順に {SHOW} つまで、残りは「以下〇個」。",
     "「その他」= その観点の上位 45 値に入らない値の割合（語板の席に入りきらない分）。番号はよく出る順の位置（小さいほどよく出る語、3 万語中）。", ""]
n_attr = []
for n, i in enumerate(pick):
    w = WORDS[i]; parts = []; cnt = 0
    for j, v in enumerate(VIEWS):
        p = B3[n, j]; vals = [(INV.get((j, x), "?"), float(p[x])) for x in range(45) if float(p[x]) >= TH]
        vals.sort(key=lambda t: -t[1]); cnt += len(vals)
        if vals:
            s = "・".join(f"{x} {q:.2f}" for x, q in vals[:SHOW]) + (f" 以下 {len(vals) - SHOW} 個" if len(vals) > SHOW else "")
            parts.append(f"  - **{v}**（その他 {float(p[45 + j]):.2f}）: {s}")
    n_attr.append(cnt)
    pos = ""
    if i in pk:
        q = PB[pk.index(i)]; top = torch.argsort(q, descending=True)[:2]
        pos = "　品詞: " + "・".join(f"{POSM['top'][int(t)]} {float(q[t]):.2f}" for t in top if float(q[t]) > 0.05)
    L.append(f"- **{w}**（{i + 1} 番、属性 {cnt} 個）{pos}")
    L += parts or ["  - （割合 0.02 以上の属性なし）"]
L[4:4] = [f"100 語の属性の数: 平均 {sum(n_attr) / len(n_attr):.1f} 個、最少 {min(n_attr)}、最多 {max(n_attr)}。", ""]
open("docs/board_attributes_100.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[:30]))
