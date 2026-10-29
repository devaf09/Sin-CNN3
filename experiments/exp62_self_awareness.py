"""実験62: 自覚グレインの第一歩 — 語板の出口の光の広がりから、「この答えは間違っているかも」を見分けられるか（確信度の較正）

着想: 2026-10-29 持ち主のアイディア ①「自分の状態を自覚するグレイン。各語板の出力地点の拡散などから、確信度や不確実性を反映する仕組み。
      新しい文章・会話を出力する際に、総合的に結論が間違っているかも、という自覚を内包する関数（短期的にはハルシネーション予防）」。
方法: 実験55・56 の材料（会話、非公開）と語板（よく出る名詞 400 語、実験58 の決まり: 平らにした目標・出口のずらし・2000 回）。
      ○○ 当て（会話の確かめ用の文で名詞を隠し、隣の動詞・形容詞の手がかりの明るさを掛けて 400 語から当てる）の 1 問ごとに、「自覚の手がかり」を測る:
        a. 1 番の割合（候補の明るさを割合にしたときの 1 番）
        b. 1 番と 2 番の差
        c. 候補の広がり（割合のエントロピー、大きいほど迷っている）
        d. 手がかりの暗さ: 1 番の候補の語板で、手がかりの値に届いた明るさ（学び 27 の「知らない意味は暗い」。暗いほど、その使い方を知らない）
        e. 語板どうしの食い違い: seed の違う 2 組の語板で、1 番の答えが同じか（同じ問いに、別々に育った 2 つの器が同じ答えを出すか）
      それぞれで「当たり・外れ」をどれだけ見分けられるか（AUROC: 0.5 = 見分けられない、1 = 完全に見分ける）。
      自覚を使う効果: 自信の高い順に答え、低いものは「わかりません」とするとき、答えた分の正解率がどう上がるか。
結果の置き場: results/exp62.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp62_self_awareness.py
"""
import collections, json, math, os, runpy, sys, time

import torch

sys.argv = [sys.argv[0], "--seeds", "0"]
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "exp56_goban_grow.py"))       # 材料（会話の数の表）と名詞を借りる
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t00 = time.time()
NTR, NOUNS, NVALS, NVIEW, NV, prior, FREQ = ns["NTR"], ns["NOUNS"], ns["NVALS"], ns["NVIEW"], ns["NV"], ns["prior"], ns["FREQ"]
NOUNS = NOUNS[:400]; T = NTR[FREQ]
STEP = 9


def shift(T):
    out = T.clone()
    for v in range(T.shape[1]):
        out[:, v, [(r + STEP * v) % 47 for r in range(47)]] = T[:, v, :47]
    return out


T = shift(T); tg = (T + 5 * prior(shift(NTR))[None]) / (T.sum(2, keepdim=True) + 5)
pos = lambda v, x: (NVALS[v].index(x) + STEP * NVIEW.index(v)) % 47 if x in NVALS[v][:-1] else NV - 1 + NVIEW.index(v)
Ws = []
for seed in (0, 1):
    P = train(tg, torch.ones(tg.shape[:2], dtype=torch.bool), 2000, seed, lr=0.03)
    Ws.append(run(P, len(NVIEW), tg.shape[2]))
    print("語板 seed", seed, f"{time.time() - t00:.0f} 秒", flush=True)
NI = {n: i for i, n in enumerate(NOUNS)}; tot = T.sum((1, 2)) + 1
qs = []
for line in open("D:/AI_git/Sin-CNN3_private/data/views/sentences_test.jsonl", encoding="utf-8"):
    d = json.loads(line); by = collections.defaultdict(list)
    for i, v, x in d["hits"]:
        by[i].append((NVIEW.index(v), pos(v, x)))
    for i, cl in by.items():
        if d["toks"][i][1] in NI:
            qs.append((NI[d["toks"][i][1]], cl))
print("問題", len(qs), flush=True)


def answer(W, cl):
    Pr = prob(W); sc = torch.log(tot) + sum(torch.log(Pr[:, v, x].clamp(min=1e-6)) for v, x in cl)
    p = torch.softmax(sc, 0); top = torch.topk(p, 2)
    dark = float(sum(W[top.indices[0], v, x] for v, x in cl) / len(cl))                       # 1 番の語板で、手がかりの値に届いた明るさ
    return int(top.indices[0]), float(top.values[0]), float(top.values[0] - top.values[1]), float(-(p * torch.log(p.clamp(min=1e-12))).sum()), dark


rows = []
for ans, cl in qs:
    a0 = answer(Ws[0], cl); a1 = answer(Ws[1], cl)
    rows.append(dict(ok=a0[0] == ans, p1=a0[1], margin=a0[2], ent=a0[3], bright=a0[4], agree=float(a0[0] == a1[0])))


def auroc(score, ok):
    """score が高いほど「当たり」と言っているとき、当たりと外れを見分ける力（0.5 = 見分けられない）。"""
    pos_ = [s for s, o in zip(score, ok) if o]; neg = [s for s, o in zip(score, ok) if not o]
    allv = sorted([(s, 1) for s in pos_] + [(s, 0) for s in neg]); rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            rank.setdefault(allv[k][0], (i + j + 1) / 2)
        i = j
    return (sum(rank[s] for s in pos_) - len(pos_) * (len(pos_) + 1) / 2) / max(len(pos_) * len(neg), 1)


ok = [r["ok"] for r in rows]
res = dict(n=len(rows), acc=sum(ok) / len(ok))
res["AUROC"] = {"1 番の割合": auroc([r["p1"] for r in rows], ok), "1 番と 2 番の差": auroc([r["margin"] for r in rows], ok),
                "候補の広がり（小さいほど自信）": auroc([-r["ent"] for r in rows], ok), "手がかりの明るさ": auroc([r["bright"] for r in rows], ok),
                "2 組の語板が一致": auroc([r["agree"] + 1e-3 * r["p1"] for r in rows], ok)}
order = sorted(range(len(rows)), key=lambda i: -rows[i]["p1"]); sel = {}
for frac in (0.1, 0.2, 0.5, 1.0):
    k = max(1, int(len(rows) * frac)); sel[f"自信の高い {int(frac * 100)}% だけ答える"] = sum(rows[i]["ok"] for i in order[:k]) / k
res["選んで答える"] = sel
res["2 組が一致したときの正解率"] = sum(r["ok"] for r in rows if r["agree"]) / max(1, sum(r["agree"] for r in rows))
res["2 組が食い違ったときの正解率"] = sum(r["ok"] for r in rows if not r["agree"]) / max(1, sum(1 - r["agree"] for r in rows))
res["2 組が一致した割合"] = sum(r["agree"] for r in rows) / len(rows)
res["sec"] = time.time() - t00
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
with open("results/exp62.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(exp="62", **res), ensure_ascii=False) + "\n")
