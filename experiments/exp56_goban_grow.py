"""実験56: 語板を広げる — 名詞を 1200 語に（まれな語は近い語板を写して学び始める）、動詞の語板、名詞 → 動詞 → 名詞とたどる

着想: 実験55 の続き（2026-10-28、持ち主「やったー」のあと、次の一歩として）。
  A. まれな名詞: 会話に少ししか出ない語は、数えた表がまばらで当てにならない。実験51（知らない物は近い語板を写して学び始める）を本物の会話で確かめる。
     よく出る 400 語を先に学び、401〜1200 番目の名詞（まれな語）を 3 通りで作る:
       まっさら 600 回 / まっさら 100 回 / 近い語板を写して 100 回（近さ = 学習用の数の表と、よく出る語の語板の明るさの合う率、実験51 の形）
  B. 動詞の語板: 動詞 1 語 = 1 枚。観点 = 「を・が・で・に」の名詞（例: 食べる の を → ラーメン・パン …）。実験55 の表を裏返して作る。
  C. たどる: 名詞の語板の「を」の観点 → 動詞の明るさ → 明るさを 2 乗して組み直し（実験50）→ 動詞の語板の「で」「が」の観点 → 名詞。
     例: ラーメン →（を）→ 食べる →（で）→ 店。
注意（試しで分かったこと）: 動詞の語板では、どの観点でも「その他」（上位 47 個以外の名詞）が 8 割を占める。光の量を保つ器では、
      直交した入口 v の光が同じ出口 e に届く明るさの和は Σ_v |<e, U u_v>|² ≤ |e|² = 1 なので、4 つの観点すべてで 8 割を同じ出口に集めることはできない
      （平均 25% 程度が上限）。試しでは「その他」に 33〜35% しか届かず、動詞を見ない割合（1.997 ビット）より悪かった（2.14）。
      そこで「その他」を観点ごとに別の出口にした（実験41 の、モードを増やすと光が減るのと同じ理由）。
材料: 実験55 と同じ（非公開フォルダの数の表、9 割学習・1 割確かめ）。
結果の置き場: results/exp56.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp56_goban_grow.py
"""
import argparse, collections, json, math, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="D:/AI_git/Sin-CNN3_private/data/views")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--out", default="results/exp56.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4)
NV = 48; NVIEW = ["どんな", "を", "が", "で", "に"]; VVIEW = ["を", "が", "で", "に"]
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
C = json.load(open(f"{a.data}/counts.json", encoding="utf-8"))
tot = {n: sum(sum(c.values()) for c in vs.values()) for n, vs in C["train"].items()}
NOUNS = [n for n in sorted(tot, key=tot.get, reverse=True) if n not in SKIP][:1200]
allv = {v: collections.Counter() for v in NVIEW}
for n, vs in C["train"].items():
    for v, c in vs.items():
        allv[v].update(c)
NVALS = {v: [x for x, _ in allv[v].most_common(NV - 1)] + ["その他"] for v in NVIEW}


def tab(split, words, views, vals, get):
    """(語, 観点, 値) の数。値 = 観点ごとの上位 47 個（番号 0〜46、観点どうしで同じ番号を使う）+ 観点ごとに別の「その他」（番号 47 + 観点の番号）。
    「その他」を観点ごとに分けるのは、光の量を保つ器では、直交した入口（観点）の光を同じ 1 つの出口に大きく集められないため（下の注意）。"""
    T = torch.zeros(len(words), len(views), NV - 1 + len(views))
    for i, w in enumerate(words):
        for j, v in enumerate(views):
            for x, c in get(split, w, v).items():
                T[i, j, vals[v].index(x) if x in vals[v][:-1] else NV - 1 + j] += c
    return T


nget = lambda s, n, v: C[s].get(n, {}).get(v, {})
NTR, NTE = tab("train", NOUNS, NVIEW, NVALS, nget), tab("test", NOUNS, NVIEW, NVALS, nget)
# 動詞の表 = 名詞の表を裏返す（動詞 → 観点 → 名詞の数）
VC = {s: collections.defaultdict(lambda: collections.defaultdict(collections.Counter)) for s in C}
for s in C:
    for n, vs in C[s].items():
        if n in SKIP:
            continue
        for v in VVIEW:
            for x, c in vs.get(v, {}).items():
                VC[s][x][v][n] += c
vtot = {x: sum(sum(c.values()) for c in vs.values()) for x, vs in VC["train"].items()}
VERBS = [x for x in sorted(vtot, key=vtot.get, reverse=True)][:300]
allvn = {v: collections.Counter() for v in VVIEW}
for x, vs in VC["train"].items():
    for v, c in vs.items():
        allvn[v].update(c)
VVALS = {v: [n for n, _ in allvn[v].most_common(NV - 1)] + ["その他"] for v in VVIEW}
vget = lambda s, x, v: VC[s].get(x, {}).get(v, {})
VTR, VTE = tab("train", VERBS, VVIEW, VVALS, vget), tab("test", VERBS, VVIEW, VVALS, vget)
norm = lambda T: T / T.sum(2, keepdim=True).clamp(min=1)
cnt = lambda T: (T + 0.5) / (T + 0.5).sum(2, keepdim=True)
prior = lambda T: (T.sum(0) + 0.5) / (T.sum(0) + 0.5).sum(1, keepdim=True)
FREQ, RARE = slice(0, 400), slice(400, 1200)
print("まれな名詞（401〜1200 番目）の学習用の数:", int(NTR[RARE].sum(2).sum(1).min()), "〜", int(NTR[RARE].sum(2).sum(1).max()), flush=True)
for seed in range(a.seeds):
    t0 = time.time(); row = dict(exp="56", seed=seed)
    # A. 名詞
    Pf = train(norm(NTR[FREQ]), NTR[FREQ].sum(2) >= 3, 600, seed)
    Wf = run(Pf, len(NVIEW), NV - 1 + len(NVIEW)); Prf = prob(Wf)
    T = NTR[RARE]; use = T.sum(2) >= 1; Ttgt = norm(T)
    lp = torch.einsum("rvx,fvx->rf", T, torch.log(Prf.clamp(min=1e-9))) / T.sum((1, 2)).clamp(min=1)[:, None]   # 合う率（対数の平均）
    near = lp.argmax(1)
    A = {}
    for kind, steps, P0 in [("まっさら 600 回", 600, None), ("まっさら 100 回", 100, None), ("近い語板を写して 100 回", 100, Pf[near]), ("近い語板を写すだけ", 0, Pf[near])]:
        P = train(Ttgt, use, steps, seed + 1, P0=P0) if steps > 0 else P0
        A[kind] = evaluate(prob(run(P, len(NVIEW), NV - 1 + len(NVIEW))), NTE[RARE])
    A["数えた表"] = evaluate(cnt(T), NTE[RARE])
    for k in (2, 5, 10):                                                                 # 比べ（強め）: 表を観点の割合へ寄せて平らに（数の少ない語ほど寄る）
        A[f"表を観点の割合で平らに k={k}"] = evaluate((T + k * prior(NTR)[None]) / (T.sum(2, keepdim=True) + k), NTE[RARE])
    A["名詞を見ない"] = evaluate(prior(NTR)[None].expand(T.shape[0], -1, -1), NTE[RARE])
    A["よく出る 400 語の語板"] = evaluate(Prf, NTE[FREQ]); A["よく出る 400 語の表"] = evaluate(cnt(NTR[FREQ]), NTE[FREQ])
    ex = {NOUNS[400 + i]: NOUNS[int(near[i])] for i in range(0, 800, 80)}
    row.update(A=A, near_examples=ex)
    print(seed, "A まれな名詞:", {k: round(v["bits"], 3) for k, v in A.items()}, "近い語板の例", ex, flush=True)
    # B. 動詞
    Pv = train(norm(VTR), VTR.sum(2) >= 3, 600, seed)
    Wv = run(Pv, len(VVIEW), NV - 1 + len(VVIEW))
    B = {"語板": evaluate(prob(Wv), VTE), "数えた表": evaluate(cnt(VTR), VTE), "表を平らに k=10": evaluate((VTR + 10 * prior(VTR)[None]) / (VTR.sum(2, keepdim=True) + 10), VTE), "動詞を見ない": evaluate(prior(VTR)[None].expand(len(VERBS), -1, -1), VTE)}
    row["B"] = B
    print(seed, "B 動詞:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in B.items()}, flush=True)
    # C. たどる: 名詞 →（を）→ 動詞 →（で / が）→ 名詞
    Prv = prob(Wv); chain = {}
    for n in ["ラーメン", "映画", "電車", "服", "本", "ゲーム", "料理", "ビール"]:
        if n not in NOUNS[:400]:
            continue
        d = Prf[NOUNS.index(n), NVIEW.index("を")]                                         # を の観点の動詞の割合
        vs = [x for x in NVALS["を"][:-1] if x in VERBS]
        w = torch.tensor([float(d[NVALS["を"].index(x)]) for x in vs]) ** 2; w = w / w.sum()   # 明るさを 2 乗して組み直す（実験50）
        top_v = [vs[int(j)] for j in torch.argsort(w, descending=True)[:3]]
        out = {}
        for v2 in ["で", "が"]:
            mix = sum(w[i] * Prv[VERBS.index(x), VVIEW.index(v2)] for i, x in enumerate(vs))
            out[v2] = [VVALS[v2][int(j)] for j in torch.argsort(mix[:NV - 1], descending=True)[:5]]
        chain[n] = dict(verbs=top_v, **out)
    row["C"] = chain; row["sec"] = time.time() - t0
    print(seed, "C たどる:", chain, f"{row['sec']:.0f} 秒", flush=True)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
