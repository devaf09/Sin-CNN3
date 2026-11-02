"""実験112: 往復（混ぜた語 → 並べ直し）は言語に依らないか。英語と日本語で同じ仕組みを回す（持ち主「この方法なら日本語以外も学べる気がする」）

仕組み（言語の知識を入れない）:
  単位 = 語（英語は空白・記号で切り、小文字にする。日本語は Sudachi の語。句読点・記号は外す）。※ 語の切り方だけは言語ごと（次の段で学ぶ）。
  2 つの語 x・y の「x が y より前」の見込み = σ(θ[型_x, 型_y])。型 = よく出る K 語はその語そのもの、ほかは「出る回数の帯」（まれな語ほど中身の語）。
  θ を、教えの文の本当の並び（2 つずつの前後）から学ぶ（実験111 と同じ。答えは元の文そのもの）。
  並べ直し: 混ぜた語のすべての並び（7 語以下）から、見込みの log の合計が一番大きいもの。
比べ: K = 50・200・1000（覚える語の数）、でたらめ。英語と日本語で同じ数で比べる。
材料: Tatoeba の英語（CC BY 2.0 FR、20 万文から）と日本語（tatoeba_jpn、CC BY 2.0 FR）。教え 10 万文、試し 4〜7 語の文 300。
  表示: 文は Tatoeba（tatoeba.org）のもの、CC BY 2.0 FR。
結果: results/exp112.txt、docs/reverse_use.md に追記、図 docs/figs/round_trip_multilingual_112.svg。実行: python experiments/exp112_round_trip_multilingual.py
"""
import itertools, json, math, random, re, sys
from collections import Counter

import torch

sys.stdout.reconfigure(encoding="utf-8"); torch.manual_seed(0); PRIV = "D:/AI_git/Sin-CNN3_private"


def load_eng():
    out = []
    for l in open(f"{PRIV}/data/text/tatoeba_eng.jsonl", encoding="utf-8"):
        ws = re.findall(r"[a-z0-9']+", json.loads(l)["text"].lower())
        if ws:
            out.append(ws)
    return out


def load_jpn():
    from sudachipy import Dictionary
    tok = Dictionary(dict="core").tokenizer(); out = []
    lines = [x for l in open(f"{PRIV}/data/text/tatoeba_jpn.jsonl", encoding="utf-8") for x in json.loads(l)["text"].split("\n") if x.strip()]   # 1 行に複数の文がまとめてある
    random.Random(5).shuffle(lines)
    for x in lines[:125000]:
        ws = [m.surface() for m in tok.tokenize(x) if not m.part_of_speech()[0].startswith(("補助記号", "空白"))]
        if ws:
            out.append(ws)
    return out


def run(lang, S):
    rng = random.Random(0); rng.shuffle(S)
    test = [s for s in S[:20000] if 4 <= len(s) <= 7][:300]; train = S[20000:120000]
    freq = Counter(w for s in train for w in s); ranked = [w for w, _ in freq.most_common()]
    out = []
    for K in (50, 200, 1000):
        top = {w: i for i, w in enumerate(ranked[:K])}
        band = lambda w: K + min(int(math.log2(freq.get(w, 0) + 1)), 15)                        # まれさの帯（16 通り）
        typ = lambda w: top.get(w, band(w)); NTY = K + 16
        X, Y = [], []
        for s in train:
            t = [typ(w) for w in s]
            for i in range(len(t)):
                for j in range(i + 1, min(len(t), i + 8)):
                    X += [(t[i], t[j]), (t[j], t[i])]; Y += [1.0, 0.0]
        X = torch.tensor(X); Y = torch.tensor(Y)
        th = torch.zeros(NTY, NTY, requires_grad=True); opt = torch.optim.Adam([th], lr=0.05)
        for ep in range(3):
            perm = torch.randperm(len(Y))
            for i in range(0, len(perm), 16384):
                idx = perm[i:i + 16384]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(th[X[idx, 0], X[idx, 1]], Y[idx]) + 1e-5 * th.pow(2).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        th = th.detach(); ok = pok = ptot = 0; rnd = 0.0
        for s in test:
            n = len(s); t = [typ(w) for w in s]
            lp = [[-math.log1p(math.exp(-float(th[t[x], t[y]]))) if x != y else 0 for y in range(n)] for x in range(n)]
            best = max(itertools.permutations(range(n)), key=lambda p: sum(lp[p[a]][p[c]] for a in range(n) for c in range(a + 1, n)))
            ok += [s[i] for i in best] == s; rnd += 1 / math.factorial(n)                         # 同じ語が 2 回ある文は、語の並びで比べる
            pos = {p: i for i, p in enumerate(best)}
            for x in range(n):
                for y in range(x + 1, n):
                    if s[x] != s[y]:
                        pok += pos[x] < pos[y]; ptot += 1
        ex = rng.choice(test); t = [typ(w) for w in ex]; n = len(ex)
        lp = [[-math.log1p(math.exp(-float(th[t[x], t[y]]))) if x != y else 0 for y in range(n)] for x in range(n)]
        shuf = list(range(n)); random.Random(1).shuffle(shuf)
        best = max(itertools.permutations(range(n)), key=lambda p: sum(lp[p[a]][p[c]] for a in range(n) for c in range(a + 1, n)))
        r = dict(lang=lang, K=K, exact=ok / len(test), pairwise=pok / ptot, random_exact=rnd / len(test),
                 example=dict(original=" ".join(ex), shuffled=" ".join(ex[i] for i in shuf), reordered=" ".join(ex[i] for i in best)))
        out.append(r); print(lang, "K", K, "丸ごと", round(r["exact"], 3), "2 つずつ", round(r["pairwise"], 3), "でたらめ", round(r["random_exact"], 3), r["example"], flush=True)
    return out


res = dict(exp="112", runs=run("英語", load_eng()) + run("日本語", load_jpn()))
with open("results/exp112.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
