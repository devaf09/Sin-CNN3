"""実験108: 共鳴パーシングの続き（実験107: 答えなしの自己学習は すぐ右 67% から抜け出せず、答えありでも 手の決まり 75% 止まり）

足すもの:
  1. 遠さの学び: 尾ごとに「係り先の鎖の何番目に係るか」（0・1・2・3 以上）と「文の最後の文節に係るか」の割合を学ぶ。
     読点（が、）や は は遠くへ、の は すぐ右へ、という性質を、決まりではなく数えて持つ。点 = log P(型_j | 尾_i) + log P(何番目・最後か | 尾_i)。
  2. やわらかい自己学習: 自分の解析の 1 本の木（硬い）ではなく、鎖の上の各候補に「その候補に係る見込み」を配って数え直す（やわらかい）。
     最初は 鎖の候補に 1/(何番目+1) を配るだけ（近いほど少し重い、答えなし）。
  3. 少しだけ答え: 教えの文のうち 100・300・1000 文だけ GiNZA の係り先で数え、残りは自分の解析で数え直す（半分教えあり）。
比べ: 実験107（すぐ右 67.4%、手の決まり 75.2%、答えあり 74.7%）。
材料・評価は実験107 と同じ（会話・青空文庫 1.5 万文、試し 1500 文、係り先の正答）。
結果: results/exp108.txt、docs/resonance_parser_107.md に追記、図 docs/figs/resonance_parser_108.svg。実行: python experiments/exp108_resonance_parser2.py
"""
import json, math, random, sys
from collections import Counter, defaultdict

src = open("experiments/exp107_resonance_parser.py", encoding="utf-8").read().split("res = dict(")[0]
exec(src)                                                                                      # 材料・尾と型・鎖・手の決まり を実験107 から


def bucket(r, last):
    return (min(r, 3), last)


class Model:
    """共鳴 P(型 | 尾) と 遠さ P(何番目・最後か | 尾) の数え上げ。"""
    def __init__(self):
        self.c = Counter(); self.ct = Counter(); self.d = Counter(); self.dt = Counter(); self.types = set()

    def add(self, bi, bj, r, last, w):
        t = bi["tail"]; self.c[(t, bj["cpos"])] += w; self.ct[t] += w; self.types.add(bj["cpos"])
        self.d[(t, bucket(r, last))] += w; self.dt[t] += w

    def score(self, bi, bj, r, last, use_dist=True):
        t = bi["tail"]; s = math.log((self.c[(t, bj["cpos"])] + 0.1) / (self.ct[t] + 0.1 * len(self.types)))
        if use_dist:
            s += math.log((self.d[(t, bucket(r, last))] + 0.1) / (self.dt[t] + 0.8))
        return s


def parse2(s, m, use_dist=True, alpha=0.0):
    n = len(s["b"]); heads = [None] * n
    for i in range(n - 2, -1, -1):
        cand = chain(heads, i, n)
        heads[i] = max(((m.score(s["b"][i], s["b"][j], r, j == n - 1, use_dist) - alpha * math.log(1 + r), j) for r, j in enumerate(cand)))[1]
    heads[n - 1] = n - 1
    return heads


def uas2(data, m, use_dist=True, alpha=0.0):
    ok = tot = 0
    for s in data:
        h = parse2(s, m, use_dist, alpha)
        for i, b in enumerate(s["b"][:-1]):
            if not b.get("skip"):
                ok += h[i] == b["head"]; tot += 1
    return ok / tot


def count_gold(data, m):
    for s in data:
        n = len(s["b"]); heads = [b["head"] if b["head"] > i else n - 1 for i, b in enumerate(s["b"])]   # 左や自分に係る所（まれ）は文末へ（鎖が回らないように）
        heads[n - 1] = None
        for i, b in enumerate(s["b"][:-1]):
            if b.get("skip"):
                continue
            cand = chain(heads, i, n)                                                         # 答えの木の上での鎖
            r = cand.index(b["head"]) if b["head"] in cand else min(len(cand) - 1, 3)
            m.add(b, s["b"][b["head"]], r, b["head"] == n - 1, 1.0)


def count_hard(data, m_old, m):
    for s in data:
        n = len(s["b"]); h = parse2(s, m_old)
        for i in range(n - 1):
            cand = chain(h, i, n); m.add(s["b"][i], s["b"][h[i]], cand.index(h[i]), h[i] == n - 1, 1.0)


def count_soft(data, m_old, m, temp=1.0, harmonic=False):
    """鎖の上の候補に、係る見込みを配って数える（やわらかい）。鎖は今の解析の木から。"""
    for s in data:
        n = len(s["b"]); h = parse2(s, m_old) if not harmonic else [min(i + 1, n - 1) for i in range(n)]
        for i in range(n - 1):
            cand = chain(h, i, n)
            if harmonic:
                ws = [1.0 / (r + 1) for r in range(len(cand))]
            else:
                ws = [math.exp(m_old.score(s["b"][i], s["b"][j], r, j == n - 1) / temp) for r, j in enumerate(cand)]
            z = sum(ws)
            for r, (j, w) in enumerate(zip(cand, ws)):
                m.add(s["b"][i], s["b"][j], r, j == n - 1, w / z)


res = dict(exp="108", runs=[])
log = lambda name, v, **kw: (res["runs"].append(dict(name=name, uas=v, **kw)), print(name, round(v, 4), kw, flush=True))
g = Model(); count_gold(tr, g)
log("答えあり 共鳴だけ", uas2(te, g, use_dist=False, alpha=0.3))
log("答えあり 共鳴 + 遠さ", uas2(te, g))
# 遠さの学びの中身（尾ごとに、すぐ右・2 番目・3 番目・4 番目以上・文の最後 の割合）
res["dist_examples"] = {t: {f"{k[0]}{'・最後' if k[1] else ''}": round(g.d[(t, k)] / g.dt[t], 2) for k in sorted({k for (tt, k) in g.d if tt == t}) if g.d[(t, k)] / g.dt[t] >= 0.05}
                        for t in ["の", "が", "が、", "は", "は、", "を", "て", "て、", "∅形容詞", "∅副詞"] if g.dt[t]}
print(json.dumps(res["dist_examples"], ensure_ascii=False), flush=True)
# 答えなし: 硬い（すぐ右から）・やわらかい（1/(何番目+1) から）
for kind in ("硬い", "やわらかい"):
    m = Model()
    if kind == "硬い":
        for s in tr:
            for i in range(len(s["b"]) - 1):
                m.add(s["b"][i], s["b"][i + 1], 0, i + 1 == len(s["b"]) - 1, 1.0)
    else:
        count_soft(tr, None, m, harmonic=True)
    curve = [uas2(te, m)]
    for it in range(5):
        m2 = Model(); (count_hard if kind == "硬い" else count_soft)(tr, m, m2); m = m2; curve.append(uas2(te, m))
    log(f"答えなし（{kind}自己学習、共鳴 + 遠さ）", curve[-1], curve=[round(x, 4) for x in curve])
# 少しだけ答え + 残りは自己学習
rng = random.Random(0); order = list(range(len(tr))); rng.shuffle(order)
for k in (100, 300, 1000):
    gold_part = [tr[i] for i in order[:k]]; rest = [tr[i] for i in order[k:]]
    m = Model(); count_gold(gold_part, m); first = uas2(te, m); curve = [first]
    for it in range(3):
        m2 = Model(); count_gold(gold_part, m2)
        for _ in range(3):                                                                     # 答えの文は 3 倍の重みで残す
            count_gold(gold_part, m2)
        count_hard(rest, m, m2); m = m2; curve.append(uas2(te, m))
    log(f"答え {k} 文 + 残りは自己学習", curve[-1], k=k, curve=[round(x, 4) for x in curve])
with open("results/exp108.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
