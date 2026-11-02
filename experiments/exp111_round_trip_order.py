"""実験111: 往復（文 → 混ぜた袋 → 並べ直した文）で、並びの重みを学ぶ（docs/reverse_use.md の 6、持ち主「どんどん進めて OK」）

考え: 文の文節を混ぜて袋にし、並べ直して元の文に戻るかを物差しにする。答えは元の文そのもの（人の木は要らない、言語に依らない）。
  並びの重み: 2 つの文節 x・y について「x が y より前に来る」見込み = σ(θ[尾_x, 尾_y] + θ[品詞_x, 品詞_y] + λ × 共鳴(x, y) の向きの項)。
             尾・品詞の組の重みを、教えの文の本当の並び（2 つずつの前後）から学ぶ（ロジスティック回帰、勾配）。
  並べ直し: 混ぜた文節のすべての並び（6 文節以下）の中で、2 つずつの見込みの log の合計が一番大きいものを選ぶ。
比べ: でたらめ、実験110 の逆回し（穴埋めの重みを逆に回す、丸ごと 16%・2 つずつ 60%）。
  さらに「尾だけ」「品詞だけ」「尾 + 品詞」で、何が並びを決めているかを見る。
材料: 実験107 と同じ（会話・青空文庫 1.5 万文）。試し: 3〜6 文節の文 300（実験110 と同じ選び方）。
結果: results/exp111.txt、docs/reverse_use.md に追記、図 docs/figs/round_trip_111.svg。実行: python experiments/exp111_round_trip_order.py
"""
import itertools, json, math, random, sys
from collections import Counter

import torch

src = open("experiments/exp107_resonance_parser.py", encoding="utf-8").read().split("res = dict(")[0]
exec(src)                                                                                      # 材料・尾・品詞（実験107）
torch.manual_seed(0)
TAILS = [t for t, c in Counter(b["tail"] for s in tr for b in s["b"]).most_common(80)]; TI = {t: i for i, t in enumerate(TAILS)}
POS = [p for p, c in Counter(b["cpos"] for s in tr for b in s["b"]).most_common(15)]; PI = {p: i for i, p in enumerate(POS)}
NT, NP = len(TAILS) + 1, len(POS) + 1
ti = lambda b: TI.get(b["tail"], NT - 1); pi = lambda b: PI.get(b["cpos"], NP - 1)


def pairs(data):
    X, Y = [], []
    for s in data:
        B = s["b"]
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                X.append((ti(B[i]), ti(B[j]), pi(B[i]), pi(B[j]))); Y.append(1.0)                 # i が j より前（本当の並び）
                X.append((ti(B[j]), ti(B[i]), pi(B[j]), pi(B[i]))); Y.append(0.0)                 # 逆の向きは 0
    return torch.tensor(X), torch.tensor(Y)


Xtr, Ytr = pairs(tr)
print("教えの組", len(Ytr), flush=True)


def train(use_tail, use_pos):
    wt = torch.zeros(NT, NT, requires_grad=True); wp = torch.zeros(NP, NP, requires_grad=True)
    opt = torch.optim.Adam([wt, wp], lr=0.05)
    for ep in range(3):
        perm = torch.randperm(len(Ytr))
        for i in range(0, len(perm), 8192):
            idx = perm[i:i + 8192]; x = Xtr[idx]
            z = (wt[x[:, 0], x[:, 1]] if use_tail else 0) + (wp[x[:, 2], x[:, 3]] if use_pos else 0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, Ytr[idx]) + 1e-4 * (wt.pow(2).sum() + wp.pow(2).sum())
            opt.zero_grad(); loss.backward(); opt.step()
    return wt.detach(), wp.detach()


def reorder_eval(wt, wp, use_tail, use_pos, sel):
    ok = pok = ptot = 0
    for s in sel:
        B = s["b"]; n = len(B)
        lp = [[0.0] * n for _ in range(n)]
        for x in range(n):
            for y in range(n):
                if x != y:
                    z = (float(wt[ti(B[x]), ti(B[y])]) if use_tail else 0) + (float(wp[pi(B[x]), pi(B[y])]) if use_pos else 0)
                    lp[x][y] = -math.log1p(math.exp(-z))                                       # log σ(z) = x が y より前の見込み
        order = list(range(n)); random.Random(len(s["text"])).shuffle(order)                  # 混ぜた袋（並べ直しは袋の順に依らない）
        best = max(itertools.permutations(order), key=lambda p: sum(lp[p[a]][p[c]] for a in range(n) for c in range(a + 1, n)))
        ok += list(best) == list(range(n)); pos = {p: i for i, p in enumerate(best)}
        for x in range(n):
            for y in range(x + 1, n):
                pok += pos[x] < pos[y]; ptot += 1
    return ok / len(sel), pok / ptot


sel = [s for s in te if 3 <= len(s["b"]) <= 6][:300]
res = dict(exp="111", n_test=len(sel), runs=[])
for name, ut, up in (("尾だけ", True, False), ("品詞だけ", False, True), ("尾 + 品詞", True, True)):
    wt, wp = train(ut, up); ex, pw = reorder_eval(wt, wp, ut, up, sel)
    res["runs"].append(dict(name=name, exact=ex, pairwise=pw)); print(name, "丸ごと", round(ex, 3), "2 つずつ", round(pw, 3), flush=True)
    if ut and up:
        WT = wt
# 学んだ並びの例: 尾ごとに、ほかの尾より前に来やすいか（行の平均）
lead = {t: round(float(WT[TI[t]].mean() - WT[:, TI[t]].mean()), 2) for t in ["は", "が", "を", "に", "で", "の", "て", "た", "ます", "∅副詞", "∅連体詞", "∅感動詞", "です", "か"] if t in TI}
res["lead"] = dict(sorted(lead.items(), key=lambda kv: -kv[1])); print(res["lead"])
res["random_exact"] = sum(1 / math.factorial(len(s["b"])) for s in sel) / len(sel)
with open("results/exp111.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
