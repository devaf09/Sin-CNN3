"""実験60: 材料の混ぜ方（分野の偏り）— 会話に強い語板にするには、どの材料をどれだけ混ぜるか（表だけで先に調べる）

着想: 実験59 で、会話の確かめ用では「会話だけで数えた表」（1.97 ビット）が「全部まとめた表」（2.42）よりずっと良かった。
      全部まとめると、ウィキペディア（観点の出現の約 9 割）の使い方に引っ張られる（例: を 見る → 日の目・憂き目）。
      語板は表をまねて学ぶので、まず「どう混ぜた表が会話に強いか」を表だけで調べ、良い混ぜ方で語板を作る。
方法: 名詞は実験59 と同じ 2 万語（全部の材料で多い順）。材料ごとの学習用の数 T_s から、混ぜた数 Σ w_s T_s を作り、観点ごとの割合へ寄せて平らにする（k = 5）。
      確かめ = 会話（実雑談・旅行の相談）の確かめ用、と全部の材料の確かめ用。値の置き方（上位 47 + その他）は、全部の材料で 1 回だけ決める（混ぜ方ごとに変えると「その他」の中身が変わり、比べられない。試しで気づいた）。
      混ぜ方: 会話だけ / 全部（重み 1）/ 会話の重みを 3・10・30・100 / 会話 × 30 + 小説（ある場合）など。
結果の置き場: results/exp60.txt
実行: python experiments/exp60_domain_mix.py
"""
import argparse, collections, json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="D:/AI_git/Sin-CNN3_private/data/views")
ap.add_argument("--nouns", type=int, default=20000)
ap.add_argument("--out", default="results/exp60.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); t00 = time.time()
NVIEW, NV = ["どんな", "を", "が", "で", "に"], 48
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
SRC = {"会話": ["conv"], "青空文庫": ["aozora_full"], "Tatoeba": ["tatoeba"], "ウィキペディア": ["wiki0", "wiki1", "wiki2"],
       "Web 小説": [f"novels_{i}" for i in range(7)]}
SRC = {k: [p for p in v if os.path.exists(f"{a.data}/counts_{p}.json")] for k, v in SRC.items()}
SRC = {k: v for k, v in SRC.items() if v}
print("材料", SRC, flush=True)
A = json.load(open(f"{a.data}/counts_all.json", encoding="utf-8"))
tot = {n: sum(sum(c.values()) for c in vs.values()) for n, vs in A["train"].items()}
NOUNS = [n for n in sorted(tot, key=tot.get, reverse=True) if n not in SKIP][:a.nouns]; NI = {n: i for i, n in enumerate(NOUNS)}
del A
# 材料ごとに、(名詞, 観点, 値の語) の数を疎な形で持つ
raw = {}
for k, parts in SRC.items():
    for split in ("train", "test"):
        cnt = collections.Counter()
        for p in parts:
            C = json.load(open(f"{a.data}/counts_{p}.json", encoding="utf-8"))[split]
            for n, vs in C.items():
                if n in NI:
                    for v, c in vs.items():
                        for x, m in c.items():
                            cnt[(NI[n], v, x)] += m
        raw[(k, split)] = cnt
    print(k, "読んだ", f"{time.time() - t00:.0f} 秒", flush=True)


def tables(weights):
    """混ぜた学習用の数で値の置き方を決め、学習用（混ぜた）・確かめ用（会話・全部）の表を作る。"""
    mix = collections.Counter()
    for k, w in weights.items():
        for key, m in raw[(k, "train")].items():
            mix[key] += w * m
    pos = POS

    def T_of(counter):
        T = torch.zeros(len(NOUNS), len(NVIEW), NV - 1 + len(NVIEW))
        for (i, v, x), m in counter.items():
            j = NVIEW.index(v); T[i, j, pos[v].get(x, NV - 1 + j)] += m
        return T
    alltest = collections.Counter()
    for k in SRC:
        alltest.update(raw[(k, "test")])
    return T_of(mix), T_of(raw[("会話", "test")]), T_of(alltest)


al = {v: collections.Counter() for v in NVIEW}                                              # 値の置き方は全部の材料（重み 1）で 1 回だけ決める（混ぜ方どうしで比べられるように）
for k in SRC:
    for (i, v, x), m in raw[(k, "train")].items():
        al[v][x] += m
POS = {v: {x: (r + 9 * j) % 47 for r, (x, _) in enumerate(al[v].most_common(NV - 1))} for j, v in enumerate(NVIEW)}
pri = lambda T: (T.sum(0) + 0.5) / (T.sum(0) + 0.5).sum(1, keepdim=True)
MIXES = {"会話だけ": {"会話": 1}, "全部（重み 1）": {k: 1 for k in SRC}}
for wc in (3, 10, 30, 100):
    MIXES[f"会話 × {wc} + ほか"] = {k: (wc if k == "会話" else 1) for k in SRC}
if "Web 小説" in SRC:
    MIXES["会話 × 30 + 小説（ウィキペディアなし）"] = {k: (30 if k == "会話" else 1) for k in SRC if k != "ウィキペディア"}
    MIXES["会話 × 30 + 小説 + ウィキペディア × 0.1"] = {k: (30 if k == "会話" else 0.1 if k == "ウィキペディア" else 1) for k in SRC}
res = {}
for name, w in MIXES.items():
    TR, TEc, TEa = tables(w)
    for k in (2, 5, 20):
        tg = (TR + k * pri(TR)[None]) / (TR.sum(2, keepdim=True) + k)
        res[f"{name} k={k}"] = dict(会話=evaluate(tg, TEc)["bits"], 全部=evaluate(tg, TEa)["bits"])
    print(name, {kk: {a2: round(b2, 3) for a2, b2 in v.items()} for kk, v in res.items() if kk.startswith(name)}, flush=True)
with open(a.out, "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(exp="60", sources=SRC, nouns=len(NOUNS), res=res, sec=time.time() - t00), ensure_ascii=False) + "\n")
