"""実験69: テストボット01 用の語板を作る — 会話の言い回し（食べる・好き・行く…）を値に持つ、会話に強い名詞 5000 語・動詞 1000 語

着想: 2026-10-29 持ち主「現状の学習結果から会話ボットを組み立てられる？ 次の試験で使いたい、テストボット01 として」。
      実験59・61 の語板は、値（観点ごとの上位 47 個）を全部の材料の多い順で決めていたので、ウィキペディア・小説の言い回し（行う・務める…）が多く、
      会話でよく使う「食べる」「好き」が値に入っていなかった。→ 値も会話を 100 倍に数えた混ぜ方で決め直して、会話向けの語板を作る。
方法: 材料 = 会話（× 100）+ 青空文庫の本文全体 + Tatoeba + ウィキペディア + Web 小説（非公開、研究のみ）。
      名詞 = 混ぜた数の多い順に 5000 語（形式名詞を除く）、値 = 観点ごとに混ぜた数の多い順に 47 個 + 観点ごとの「その他」（出口は実験58 のずらし）。
      動詞 = 混ぜた数の多い順に 1000 語、観点 = を・が・で・に の名詞（値 = 観点ごとに 47 個 + その他）。
      目標 = 平らにした表（k = 20）、2000 回・学習率 0.03、2 値の初期板（実験58 の決まり）。
出力: 非公開フォルダの models/testbot01.pt（板・語・値の並び）。結果の数字: results/exp69.txt
実行: python experiments/exp69_testbot_model.py --device cuda（experiments/sweeps/sweep69.sh）
"""
import argparse, collections, json, os, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="D:/AI_git/Sin-CNN3_private/data/views")
ap.add_argument("--out_model", default="D:/AI_git/Sin-CNN3_private/models/testbot01.pt")
ap.add_argument("--nouns", type=int, default=5000)
ap.add_argument("--verbs", type=int, default=1000)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--device", default="cpu")
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

sys.stdout.reconfigure(encoding="utf-8"); t00 = time.time()
NVIEW, VVIEW, NV = ["どんな", "を", "が", "で", "に"], ["を", "が", "で", "に"], 48
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
SRC = {"conv": 100, "aozora_full": 1, "tatoeba": 1, "wiki0": 1, "wiki1": 1, "wiki2": 1}
SRC.update({f"novels_{i}": 1 for i in range(7)})
mix = {s: collections.Counter() for s in ("train", "test")}                                  # (名詞, 観点, 値) → 混ぜた数
conv_test = collections.Counter()
for name, w in SRC.items():
    p = f"{a.data}/counts_{name}.json"
    if not os.path.exists(p):
        continue
    C = json.load(open(p, encoding="utf-8"))
    for s in mix:
        for n, vs in C[s].items():
            if n in SKIP:
                continue
            for v, c in vs.items():
                for x, m in c.items():
                    mix[s][(n, v, x)] += w * m
                    if name == "conv" and s == "test":
                        conv_test[(n, v, x)] += m
    print(name, "読んだ", f"{time.time() - t00:.0f} 秒", flush=True)
ntot = collections.Counter(); vtot = collections.Counter(); nval = {v: collections.Counter() for v in NVIEW}; vval = {v: collections.Counter() for v in VVIEW}
for (n, v, x), m in mix["train"].items():
    ntot[n] += m; nval[v][x] += m
    if v in VVIEW:
        vtot[x] += m; vval[v][n] += m
NOUNS = [n for n, _ in ntot.most_common(a.nouns)]; VERBS = [x for x, _ in vtot.most_common(a.verbs)]
NVALS = {v: [x for x, _ in nval[v].most_common(NV - 1)] for v in NVIEW}; VVALS = {v: [x for x, _ in vval[v].most_common(NV - 1)] for v in VVIEW}
NPOS = {v: {x: (r + 9 * j) % 47 for r, x in enumerate(NVALS[v])} for j, v in enumerate(NVIEW)}
VPOS = {v: {x: (r + 9 * j) % 47 for r, x in enumerate(VVALS[v])} for j, v in enumerate(VVIEW)}
NI = {n: i for i, n in enumerate(NOUNS)}; VI = {x: i for i, x in enumerate(VERBS)}


def tables(counter):
    """混ぜた数から、名詞の表 (名詞, 観点, 値) と動詞の表 (動詞, 観点, 名詞) を作る。"""
    TN = torch.zeros(len(NOUNS), len(NVIEW), NV - 1 + len(NVIEW)); TV = torch.zeros(len(VERBS), len(VVIEW), NV - 1 + len(VVIEW))
    for (n, v, x), m in counter.items():
        j = NVIEW.index(v)
        if n in NI:
            TN[NI[n], j, NPOS[v].get(x, NV - 1 + j)] += m
        if v in VVIEW and x in VI:
            k = VVIEW.index(v); TV[VI[x], k, VPOS[v].get(n, NV - 1 + k)] += m
    return TN, TV


TN, TV = tables(mix["train"]); TNe, TVe = tables(conv_test)
pri = lambda T: (T.sum(0) + 0.5) / (T.sum(0) + 0.5).sum(1, keepdim=True)
sm = lambda T: (T + 20 * pri(T)[None]) / (T.sum(2, keepdim=True) + 20)
print("表を作った", f"{time.time() - t00:.0f} 秒", flush=True)
res = dict(exp="69", nouns=len(NOUNS), verbs=len(VERBS))
out = dict(NVIEW=NVIEW, VVIEW=VVIEW, NOUNS=NOUNS, VERBS=VERBS, NPOS=NPOS, VPOS=VPOS, noun_count={n: ntot[n] for n in NOUNS}, verb_count={x: vtot[x] for x in VERBS})
for name, T, Te, nv in (("名詞", TN, TNe, len(NVIEW)), ("動詞", TV, TVe, len(VVIEW))):
    t0 = time.time(); tg = sm(T); Ps = []
    for i in range(0, len(tg), 2500):
        part = tg[i:i + 2500]; Ps.append(train(part, torch.ones(part.shape[:2], dtype=torch.bool), a.steps, i, lr=0.03, dev=a.device).cpu())
    P = torch.cat(Ps); W = torch.cat([run(P[i:i + 2500].to(a.device), nv, tg.shape[2]).cpu() for i in range(0, len(P), 2500)])
    out[f"P_{name}"] = P
    res[name] = dict(語板_会話の確かめ用=evaluate(prob(W), Te)["bits"], 表_会話の確かめ用=evaluate(tg, Te)["bits"], 光=float(W.sum(-1).mean()), sec=time.time() - t0)
    print(name, res[name], flush=True)
os.makedirs(os.path.dirname(a.out_model), exist_ok=True); torch.save(out, a.out_model)
res["sec"] = time.time() - t00
with open("results/exp69.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print("保存した", a.out_model, "を の値:", NVALS["を"][:20])
