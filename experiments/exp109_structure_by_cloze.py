"""実験109: 構造の良し悪しを「何が当たるようになるか」で測る（持ち主の振り返り 2026-11-02「文法解析というロジスティクスな解決法にも問題があった」→「いいね！！」）

物差し: 文の中の 1 つの文節の中身の語を隠し、残りの文節の中身の語から当てる（穴埋め）。当て方は 一緒に出る割合（PMI）の重ね合わせ。
        どの文節の手がかりをどれだけ重く見るか、を「構造」が決める。構造が役に立つなら、穴埋めが当たるようになる。
比べる構造（どれも同じ穴埋めの仕組みで、重みだけが違う）:
  袋: 全部同じ重み。
  並びの近さ: 文の中で何文節離れているか（0.5^(離れ − 1)）。
  GiNZA の木: 木の上で何段離れているか（0.5^(段 − 1)）。人が作った文法の木（比べのための目安）。
  自分の木（答えなし）: 実験108 の答えなしの自己学習で張った木。
  自分の木（答えあり + 遠さ）: 実験108 の 76.6% の木。
  （木が当たるほど穴埋めが当たるのか、それとも並びの近さで十分なのか、を見る）
評価: 試しの文の各文節を 1 つずつ隠し、正しい語の順位の逆数の平均（MRR）。候補 = 教えに 5 回以上出た語。
材料: 実験107 と同じ（会話・青空文庫 1.5 万文。Web 小説は使わない）。
結果: results/exp109.txt、docs/task_yardstick.md、図 docs/figs/structure_by_cloze_109.svg。実行: python experiments/exp109_structure_by_cloze.py
"""
import json, math, sys
from collections import Counter, defaultdict

import torch

src = open("experiments/exp108_resonance_parser2.py", encoding="utf-8").read().split('res = dict(exp="108"')[0]
exec(src)                                                                                      # 材料・解析器（実験107・108）


def core_lemma(b):
    w = next((w for w in b["w"] if not w[2].startswith(FUNC)), None)
    return w[1] if w and not w[2].startswith(("補助記号", "空白")) else None


for s in S:
    for b in s["b"]:
        b["lem"] = core_lemma(b)
# ---- 一緒に出る数え上げ（教えの文、文の中の中身の語どうし）----
cnt = Counter(c for s in tr for c in {b["lem"] for b in s["b"] if b["lem"]})
V = [w for w, c in cnt.items() if c >= 5]; VI = {w: i for i, w in enumerate(V)}
co = torch.zeros(len(V), len(V))
for s in tr:
    ids = sorted({VI[b["lem"]] for b in s["b"] if b["lem"] in VI})
    for a in ids:
        for c in ids:
            if a != c:
                co[a, c] += 1
nsent = len(tr); f = torch.tensor([cnt[w] for w in V], dtype=torch.float32)
PMI = torch.log((co + 0.1) * nsent / (f[:, None] * f[None, :])).clamp(min=0)                   # 正の PMI だけ（共鳴する組だけを光らせる）
prior = torch.log(f / f.sum())
print("語", len(V), "教えの文", nsent, flush=True)


def tree_dist(heads, n):
    """木の上の段数（向きなし）。"""
    adj = defaultdict(set)
    for i in range(n):
        h = heads[i]
        if h is not None and h != i and 0 <= h < n:
            adj[i].add(h); adj[h].add(i)
    D = [[99] * n for _ in range(n)]
    for a in range(n):
        D[a][a] = 0; q = [a]
        while q:
            x = q.pop(0)
            for y in adj[x]:
                if D[a][y] == 99:
                    D[a][y] = D[a][x] + 1; q.append(y)
    return D


m_self = Model()                                                                                # 答えなし（すぐ右から 1 回数え直した木）
for s in tr:
    for i in range(len(s["b"]) - 1):
        m_self.add(s["b"][i], s["b"][i + 1], 0, i + 1 == len(s["b"]) - 1, 1.0)
m2 = Model(); count_hard(tr, m_self, m2); m_self = m2
m_gold = Model(); count_gold(tr, m_gold)
STRUCT = ["袋", "並びの近さ", "GiNZA の木", "自分の木（答えなし）", "自分の木（答えあり + 遠さ）"]


def weights(s, kind):
    n = len(s["b"])
    if kind == "袋":
        return [[1.0] * n for _ in range(n)]
    if kind == "並びの近さ":
        return [[0.5 ** (abs(a - c) - 1) for c in range(n)] for a in range(n)]
    if kind == "GiNZA の木":
        heads = [b["head"] for b in s["b"]]
    elif kind == "自分の木（答えなし）":
        heads = parse2(s, m_self)
    else:
        heads = parse2(s, m_gold)
    D = tree_dist(heads, n)
    return [[0.5 ** (D[a][c] - 1) if D[a][c] < 99 else 0.0 for c in range(n)] for a in range(n)]


res = dict(exp="109", n_vocab=len(V), runs=[])
acc = {k: [] for k in STRUCT}; uas_of = {}
for s in te:
    n = len(s["b"])
    if n < 3:
        continue
    Ws = {k: weights(s, k) for k in STRUCT}
    for a, b in enumerate(s["b"]):
        if b["lem"] not in VI:
            continue
        ctx = [(c, VI[x["lem"]]) for c, x in enumerate(s["b"]) if c != a and x["lem"] in VI]
        if not ctx:
            continue
        for k in STRUCT:
            w = torch.tensor([Ws[k][a][c] for c, _ in ctx]); w = w / w.sum().clamp(min=1e-9)
            sc = (w[:, None] * PMI[[i for _, i in ctx]]).sum(0) + 0.1 * prior                   # 手がかりの共鳴を構造の重みで重ねる
            sc[[i for _, i in ctx]] -= 1e9                                                      # 手がかりそのものは答えにしない
            rank = int((sc > sc[VI[b["lem"]]]).sum()) + 1
            acc[k].append(1.0 / rank)
for k in STRUCT:
    v = sum(acc[k]) / len(acc[k]); res["runs"].append(dict(name=k, mrr=v, n=len(acc[k])))
    print(k, "穴埋め MRR", round(v, 4), "問", len(acc[k]), flush=True)
res["uas"] = {"自分の木（答えなし）": uas2(te, m_self), "自分の木（答えあり + 遠さ）": uas2(te, m_gold), "GiNZA の木": 1.0}
print(res["uas"])
with open("results/exp109.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
