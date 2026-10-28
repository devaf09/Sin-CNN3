"""実験59: 語板を数万語に（大規模な学習の計画、段階 3。GPU）

着想: 段階 1（実験58）で語板が平らにした表に追いついた。段階 2 で材料を増やした（非公開フォルダ: 会話・青空文庫の本文全体・Tatoeba・ウィキペディア、観点の出現 約 3900 万）。
      名詞 2 万語・動詞 3000 語の語板を作り、学習に使っていない文で確かめる。
方法: 実験58 の決まり（目標を平らにする k = 5（動詞 10）、値の出口を観点ごとにずらす、2000 回・学習率 0.03、2 値の初期板）。器は src/grain.py と同じ（縦 64、板 8 列）。
      語板は 5000 枚ずつまとめて GPU で学習。値は観点ごとに上位 47 個 + 観点ごとの「その他」。
確かめ（対話・文書の番号で分けた 1 割、学習に使っていない）:
  a. 全部の材料の確かめ用 / 会話（実雑談・旅行の相談）だけの確かめ用 で、(名詞, 観点) → 値 の交差エントロピー。比べ: 平らにした表、名詞を見ない。
     さらに、会話だけで数えた表（平らに）とも比べる（材料を増やすと会話の当たりが良くなるか）。
  b. 逆引き（値 → 名詞の上位）。
  c. 会話の確かめ用の文の ○○ 当て（隣の動詞・形容詞の手がかりの掛け算、2 万語から）。
結果の置き場: results/exp59.txt。学習済みの語板は非公開フォルダ（Sin-CNN3_private/models/）に置く。
実行: python experiments/exp59_goban_large.py --device cuda（experiments/sweeps/sweep59.sh、GPU の共通の待ち行列）
"""
import argparse, collections, json, math, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="D:/AI_git/Sin-CNN3_private/data/views")
ap.add_argument("--models", default="D:/AI_git/Sin-CNN3_private/models")
ap.add_argument("--nouns", type=int, default=20000)
ap.add_argument("--verbs", type=int, default=3000)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--chunk", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp59.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); dev = a.device; t00 = time.time()
NVIEW, VVIEW, NV = ["どんな", "を", "が", "で", "に"], ["を", "が", "で", "に"], 48
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
A = json.load(open(f"{a.data}/counts_all.json", encoding="utf-8"))
CV = json.load(open(f"{a.data}/counts_conv.json", encoding="utf-8"))
tot = {n: sum(sum(c.values()) for c in vs.values()) for n, vs in A["train"].items()}
NOUNS = [n for n in sorted(tot, key=tot.get, reverse=True) if n not in SKIP][:a.nouns]
print("名詞", len(NOUNS), "いちばんまれな語の出現", tot[NOUNS[-1]], flush=True)


def vals_of(C, words, views):
    al = {v: collections.Counter() for v in views}
    for w in words:
        for v in views:
            al[v].update(C.get(w, {}).get(v, {}))
    return {v: [x for x, _ in al[v].most_common(NV - 1)] for v in views}


def table(C, words, views, vals, step=9):
    """(語, 観点, 値) の数。観点 v の r 番目の値は出口 (r + step·v) mod 47、それ以外は観点ごとの「その他」（47 + v）。"""
    T = torch.zeros(len(words), len(views), NV - 1 + len(views)); pos = {v: {x: (r + step * j) % 47 for r, x in enumerate(vals[v])} for j, v in enumerate(views)}
    for i, w in enumerate(words):
        d = C.get(w, {})
        for j, v in enumerate(views):
            for x, c in d.get(v, {}).items():
                T[i, j, pos[v].get(x, NV - 1 + j)] += c
    return T


NVALS = vals_of(A["train"], NOUNS, NVIEW)
NTR, NTE, NTEc, NTRc = (table(C, NOUNS, NVIEW, NVALS) for C in (A["train"], A["test"], CV["test"], CV["train"]))
# 動詞の表 = 名詞の表を裏返す
inv = {s: collections.defaultdict(lambda: collections.defaultdict(collections.Counter)) for s in ("train", "test", "ctest", "ctrain")}
for s, C in (("train", A["train"]), ("test", A["test"]), ("ctest", CV["test"]), ("ctrain", CV["train"])):
    for n, vs in C.items():
        if n in SKIP:
            continue
        for v in VVIEW:
            for x, c in vs.get(v, {}).items():
                inv[s][x][v][n] += c
vtot = {x: sum(sum(c.values()) for c in vs.values()) for x, vs in inv["train"].items()}
VERBS = sorted(vtot, key=vtot.get, reverse=True)[:a.verbs]
VVALS = vals_of(inv["train"], VERBS, VVIEW)
VTR, VTE, VTEc, VTRc = (table(inv[s], VERBS, VVIEW, VVALS) for s in ("train", "test", "ctest", "ctrain"))
pri = lambda T: (T.sum(0) + 0.5) / (T.sum(0) + 0.5).sum(1, keepdim=True)
smooth = lambda T, pr, k: (T + k * pr[None]) / (T.sum(2, keepdim=True) + k)
print("表を作った", f"{time.time() - t00:.0f} 秒", flush=True)


def fit(T, k, nv):
    tg = smooth(T, pri(T), k); Ps = []
    for i in range(0, len(T), a.chunk):
        t0 = time.time(); part = tg[i:i + a.chunk]
        Ps.append(train(part, torch.ones(part.shape[:2], dtype=torch.bool), a.steps, a.seed + i, lr=0.03, dev=dev).cpu())
        print(f"  {i + len(part)} / {len(T)} 枚  {time.time() - t0:.0f} 秒", flush=True)
    P = torch.cat(Ps)
    W = torch.cat([run(P[i:i + a.chunk].to(dev), nv, T.shape[2]).cpu() for i in range(0, len(P), a.chunk)])
    return P, W, tg


row = dict(exp="59", seed=a.seed, nouns=len(NOUNS), verbs=len(VERBS), steps=a.steps, device=dev, torch=torch.__version__)
for name, TR, TE, TEc, TRc, k, nv in [("名詞", NTR, NTE, NTEc, NTRc, 5, 5), ("動詞", VTR, VTE, VTEc, VTRc, 10, 4)]:
    t0 = time.time(); P, W, tg = fit(TR, k, nv); Pr = prob(W)
    os.makedirs(a.models, exist_ok=True); torch.save(dict(P=P, words=NOUNS if name == "名詞" else VERBS, vals=NVALS if name == "名詞" else VVALS), f"{a.models}/goban_{name}_seed{a.seed}.pt")
    r = {"語板 全部の確かめ用": evaluate(Pr, TE), "平らにした表 全部": evaluate(tg, TE), "語を見ない 全部": evaluate(pri(TR)[None].expand(len(TR), -1, -1), TE),
         "語板 会話の確かめ用": evaluate(Pr, TEc), "平らにした表 会話": evaluate(tg, TEc), "会話だけの表（平らに）会話": evaluate(smooth(TRc, pri(TRc), k), TEc),
         "語を見ない 会話": evaluate(pri(TR)[None].expand(len(TR), -1, -1), TEc), "光": float(W.sum(-1).mean()), "sec": time.time() - t0}
    row[name] = r
    print(name, {kk: (round(v["bits"], 3) if isinstance(v, dict) else round(v, 3)) for kk, v in r.items()}, flush=True)
    if name == "名詞":
        rev = {}
        for v, x in [("を", "飼う"), ("が", "降る"), ("を", "食べる"), ("どんな", "美味しい"), ("を", "見る"), ("に", "乗る"), ("で", "買う"), ("を", "弾く"), ("が", "咲く"), ("を", "読む")]:
            if x in NVALS[v]:
                j = NVIEW.index(v); b = W[:, j, (NVALS[v].index(x) + 9 * j) % 47]
                rev[f"{v} {x}"] = [NOUNS[int(i)] for i in torch.argsort(b, descending=True)[:8]]
        row["逆引き"] = rev; print("逆引き", rev, flush=True)
        NPr = Pr
# ○○ 当て（会話の確かめ用の文、2 万語から）
NI = {n: i for i, n in enumerate(NOUNS)}; pos = {v: {x: (r + 9 * j) % 47 for r, x in enumerate(NVALS[v])} for j, v in enumerate(NVIEW)}
freq = torch.tensor([tot[n] for n in NOUNS], dtype=torch.float32); lpN = torch.log(NPr.clamp(min=1e-6)); lpT = torch.log(smooth(NTR, pri(NTR), 5))
res = {"語板": [0, 0], "平らにした表": [0, 0], "よく出る順だけ": [0, 0]}; n_q = 0
for line in open(f"{a.data}/sentences_test_conv.jsonl", encoding="utf-8"):
    d = json.loads(line); by = collections.defaultdict(list)
    for i, v, x in d["hits"]:
        by[i].append((NVIEW.index(v), pos[v].get(x, NV - 1 + NVIEW.index(v))))
    for i, cl in by.items():
        n = d["toks"][i][1]
        if n not in NI:
            continue
        n_q += 1; ans = NI[n]
        for key, lp in (("語板", lpN), ("平らにした表", lpT), ("よく出る順だけ", None)):
            sc = torch.log(freq) + (sum(lp[:, v, x] for v, x in cl) if lp is not None else 0)
            rk = int((sc > sc[ans]).sum()); res[key][0] += rk < 1; res[key][1] += rk < 10
row["○○ 当て"] = {k: dict(top1=v[0] / n_q, top10=v[1] / n_q) for k, v in res.items()}; row["○○ 当て の問題数"] = n_q; row["sec"] = time.time() - t00
print("○○ 当て", row["○○ 当て"], n_q, flush=True)
with open(a.out, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
