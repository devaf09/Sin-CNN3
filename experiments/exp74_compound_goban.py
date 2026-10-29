"""実験74: 合成の語板 — つなげる向きの決定（前か後ろか）と、差分で持つ合成の語板。候補の選び方 (a) 過飽和 と (b) 相互情報量 を比べる

着想: 2026-10-29 持ち主「〜の〜 の相手が多すぎる語は、前か後ろの語をくっつけて新しい語板にする。それでも多ければさらにつなげる」
      「なるべく新しい語板が属性を持っていかない方向の語とつなげる」「元の語板は残し、合成の語板は元との差分で持つ」。
      候補は実験72（過飽和の案 (a) 5714 組・相互情報量の案 (b) 3000 組）。候補を 1 語として数え直した材料（非公開フォルダの counts3_*.json）。
方法:
  1. 向きの決定（(a) の過飽和の親の語ごと）:
     前につなげる = 「X の [親]」（親の「の前」の相手の上位 3）、後ろにつなげる = 「[親] の Y」（「の後」の相手の上位 3）。
     持っていく属性 = 合成語の観点（どんな・を・が・で・に・は・と、「の」の観点は除く）の数 ÷ 親の同じ観点の数。少ない向きを選ぶ（持ち主の案）。
     ついでに、合成語の使われ方が親とどれだけ違うか（KL、ビット）も測る（違うほど、別の板にする価値がある）。
     つなげたあと: 親の数から合成語の分を引いて、親の「の前」「の後」の過飽和が解けるか。合成語自身がまだ過飽和なら、さらにつなげる候補（数だけ出す）。
  2. 合成の語板（GPU）: 語板 v2（実験73、観点 9・値 45 + その他）と同じ出口の並び。
     目標 = 合成語の表を「親の語板の割合」へ寄せて平らにする（全体の割合でなく親へ寄せる = 差分で持つ）。
     比べるもの（確かめ用の材料で、交差エントロピー・ビット）:
       親の語板そのまま（合成語を作らない）/ 親から学び直した合成の語板（差分）/ まっさらから学んだ合成の語板 / 表（親へ寄せて平ら）/ 表（全体へ寄せて平ら）
     「差分の大きさ」= 板の位相が親からどれだけ動いたか（平均の絶対値）。
  候補の集合 (a)・(b) をそれぞれ 1 本の仕事にして、GPU 2 枚で同時に（--set a / --set b）。
結果: results/exp74.txt、docs/compound_74.md（向きの例・(a)(b) の比べ）。語板: 非公開フォルダの models/goban_compound_<set>.pt
実行: python experiments/exp74_compound_goban.py --device cuda --set a（experiments/sweeps/sweep74_a.sh・b.sh）
"""
import argparse, collections, glob, json, math, os, pickle, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="a")                                                     # 候補の集合: a = 過飽和、b = 相互情報量
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--min", type=int, default=30)                                            # 合成語の学習用の数（重み付き）がこれ以上のものだけ語板にする
ap.add_argument("--out", default="results/exp74.txt")
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

sys.stdout.reconfigure(encoding="utf-8"); t00 = time.time(); dev = a.device
PRIV = "D:/AI_git/Sin-CNN3_private"
VIEWS = ["どんな", "を", "が", "で", "に", "の前", "の後", "と", "は"]; NV = 45            # 語板 v2 と同じ観点・値の数
ATTRV = ["どんな", "を", "が", "で", "に", "は", "と"]                                       # 「持っていく属性」を測る観点（「の」の観点は、つなげた相手そのものなので除く）

# ---- 合成語の数（counts3_*）を、語板 v2 と同じ重みで足し合わせる（会話 × 100、ほか × 1。Web 小説は一般だけ） ----
CC = {s: collections.defaultdict(lambda: collections.defaultdict(collections.Counter)) for s in ("train", "test")}
for p in sorted(glob.glob(f"{PRIV}/data/views/counts3_*.json")):
    w = 100.0 if p.endswith("counts3_conv.json") else 1.0                                  # 会話は 100 倍（実験60・61）
    C = json.load(open(p, encoding="utf-8"))
    for s in CC:
        for key, vs in C[s].items():
            for v, c in vs.items():
                for x, m in c.items():
                    CC[s][key][v][x] += w * m                                               # 「_数」= 合成語が出た回数
print("合成語", len(CC["train"]), f"{time.time() - t00:.0f} 秒", flush=True)
CA = json.load(open(f"{PRIV}/data/views/compound_candidates.json", encoding="utf-8"))
CAND = {"a": [tuple(x) for x in CA["a"]], "b": [tuple(x) for x in CA["b"]]}
both = set(CAND["a"]) & set(CAND["b"])                                                      # 両方の案に入る組
D = pickle.load(open(f"{PRIV}/data/views/merged2.pkl", "rb")); TRP = D["train"]             # 親の語の数（観点 9、語板 v2 の材料）
print("親の材料", f"{time.time() - t00:.0f} 秒", flush=True)


def mass(vs, views):
    """観点 views の数の合計（重み付き）。"""
    return sum(sum(vs.get(v, {}).values()) for v in views)


def kl_bits(cv, pv, views):
    """合成語の使われ方 cv が、親の使われ方 pv とどれだけ違うか（KL(合成 || 親)、観点の数で重みを付けた平均、ビット）。親を少し平らにして 0 割りを防ぐ。"""
    tot = 0.0; wsum = 0.0
    for v in views:
        c = cv.get(v, {}); n = sum(c.values())
        if n <= 0:
            continue
        pp = pv.get(v, {}); pn = sum(pp.values()); keys = set(c) | set(pp); K = len(keys) + 1
        kl = sum((m / n) * math.log2((m / n) / ((pp.get(x, 0) + 0.5) / (pn + 0.5 * K))) for x, m in c.items() if m > 0)
        tot += n * kl; wsum += n
    return tot / wsum if wsum else 0.0


def saturated(c, k=3):
    """実験72 と同じ過飽和の判定: その語自身の多い順 47 値 + その他。いちばん強い点 ≤ ほかの平均 + k × 標準偏差、または「その他」> 0.5。"""
    tot = sum(c.values())
    if tot < 50:                                                                            # 数の少ない観点は判定しない（実験72 と同じ）
        return None
    vals = [m / tot for _, m in collections.Counter(c).most_common(47)]; vals += [0.0] * (47 - len(vals))
    other = 1 - sum(vals); p1 = vals[0]; rest = vals[1:]
    m = sum(rest) / len(rest); sd = math.sqrt(sum((x - m) ** 2 for x in rest) / len(rest))
    return (p1 <= m + k * sd) or (other > 0.5)


res = dict(exp="74", set=a.set)
# ---- 1. 向きの決定（(a) の親の語ごと、--set a の仕事だけで行う） ----
if a.set == "a":
    parents = collections.defaultdict(lambda: {"前": [], "後": []})                          # 親 → 前につなげる組・後ろにつなげる組
    for x, y in CAND["a"]:
        key = f"{x}の{y}"
        if key in CC["train"]:
            if y in TRP and x in TRP.get(y, {}).get("の前", {}):
                parents[y]["前"].append(key)                                                 # 「x の [y]」= 親 y の前につなげる
            if x in TRP and y in TRP.get(x, {}).get("の後", {}):
                parents[x]["後"].append(key)                                                 # 「[x] の y」= 親 x の後ろにつなげる
    rows = []
    for n, dd in parents.items():
        if not dd["前"] or not dd["後"]:
            continue                                                                        # 両方の向きの候補がある親だけ比べる
        pm = mass(TRP[n], ATTRV)
        if pm <= 0:
            continue
        r = dict(parent=n)
        for side in ("前", "後"):
            ks = dd[side]
            r[side] = ks
            r[side + "_taken"] = sum(mass(CC["train"][k], ATTRV) for k in ks) / pm          # 持っていく属性（親の観点の数のうち、合成語へ移る割合）
            r[side + "_kl"] = sum(kl_bits(CC["train"][k], TRP[n], ATTRV) * mass(CC["train"][k], ATTRV) for k in ks) / max(1e-9, sum(mass(CC["train"][k], ATTRV) for k in ks))
            # つなげたあとの親: 合成語の分を、つないだ観点（前なら「の前」、後なら「の後」）から引く
            nv = "の前" if side == "前" else "の後"
            left = collections.Counter(TRP[n].get(nv, {}))
            for k in ks:
                partner = k.split("の")[0] if side == "前" else k.split("の", 1)[1]
                left[partner] = max(0.0, left[partner] - CC["train"][k]["_数"].get("_", 0))
            r[side + "_still"] = saturated(left)                                            # 引いたあとも、親のその観点は過飽和か
            r[side + "_chain"] = sum(1 for k in ks if saturated(CC["train"][k].get("の前" if side == "前" else "の後", {})))   # 合成語自身がまだ過飽和（さらにつなげる候補）
        r["choose"] = "前" if r["前_taken"] <= r["後_taken"] else "後"                      # 持ち主の決め方: 持っていく属性が少ない向き
        rows.append(r)
    nf = sum(r["choose"] == "前" for r in rows)
    res["向き"] = dict(parents=len(rows), front=nf, back=len(rows) - nf,
                     taken_front=sum(r["前_taken"] for r in rows) / len(rows), taken_back=sum(r["後_taken"] for r in rows) / len(rows),
                     kl_front=sum(r["前_kl"] for r in rows) / len(rows), kl_back=sum(r["後_kl"] for r in rows) / len(rows),
                     still_chosen=sum(bool(r[r["choose"] + "_still"]) for r in rows), chain_chosen=sum(r[r["choose"] + "_chain"] for r in rows),
                     kl_agree=sum((r["前_kl"] >= r["後_kl"]) == (r["choose"] == "前") for r in rows))   # 「持っていかない向き」と「親との違いが大きい向き」が一致する数
    json.dump(rows, open(f"{PRIV}/data/views/compound_direction.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("向き", res["向き"], flush=True)

# ---- 2. 合成の語板（親の語板 v2 から学び直す = 差分） ----
G = {}
for k in (0, 1):                                                                            # 語板 v2（実験73、リフレッシュ側）の 2 つの片方を合わせる
    M = torch.load(f"{PRIV}/models/goban_v2_part{k}.pt", weights_only=False)
    for i, w in enumerate(M["words"]):
        G[w] = M["P"][i]
POS = M["pos"]                                                                              # 出口の並び（値 → 出口の場所、観点ごと）は 2 つで同じ
items = []                                                                                  # (合成語, 親) の並び。親 = 語板 v2 にあり、合成語の数が十分なもの
for x, y in CAND[a.set]:
    key = f"{x}の{y}"
    if key not in CC["train"] or mass(CC["train"][key], VIEWS) < a.min:
        continue
    par = y if y in G else x if x in G else None                                            # 親は後ろの語（「学校の先生」なら先生）を先に。なければ前の語
    if par:
        items.append((key, par))
print("語板にする合成語", len(items), f"{time.time() - t00:.0f} 秒", flush=True)


def table(src, keys):
    """数を語板 v2 の出口の並びの表にする（語 × 観点 9 × 出口 54）。上位 45 値の外は観点ごとの「その他」へ。"""
    T = torch.zeros(len(keys), len(VIEWS), NV + len(VIEWS))
    for i, kk in enumerate(keys):
        dd = src.get(kk, {})
        for j, v in enumerate(VIEWS):
            for x, m in dd.get(v, {}).items():
                T[i, j, POS[v].get(x, NV + j)] += m
    return T


keys = [k for k, _ in items]; pars = [p for _, p in items]
TR, TE = table(CC["train"], keys), table(CC["test"], keys)
P0 = torch.stack([G[p] for p in pars])                                                      # 親の板（ここから学び直す = 差分）
B = 4000
Wp = torch.cat([run(P0[i:i + B].to(dev), len(VIEWS), NV + len(VIEWS)).cpu() for i in range(0, len(keys), B)])
parent_p = prob(Wp)                                                                         # 親の語板の割合（合成語を作らないときの答え）
gpri = (TR.sum(0) + 0.5) / (TR.sum(0) + 0.5).sum(1, keepdim=True)                           # 全体の割合
K = 20
tg_par = (TR + K * parent_p) / (TR.sum(2, keepdim=True) + K)                                # 目標: 親へ寄せて平ら（差分で持つ）
tg_glb = (TR + K * gpri[None]) / (TR.sum(2, keepdim=True) + K)                              # 比べ: 全体へ寄せて平ら
use = torch.ones(TR.shape[:2], dtype=torch.bool)
out = {}
for name, init in (("diff", P0), ("scratch", None)):                                         # 親から学び直す / まっさらから
    P = torch.cat([train(tg_par[i:i + B], use[i:i + B], a.steps, i, lr=0.03, P0=None if init is None else init[i:i + B], dev=dev).cpu() for i in range(0, len(keys), B)])
    W = torch.cat([run(P[i:i + B].to(dev), len(VIEWS), NV + len(VIEWS)).cpu() for i in range(0, len(keys), B)])
    out[name] = evaluate(prob(W), TE)
    if name == "diff":
        Pd = P
        out["diff"]["phase_move"] = float(((P - P0 + math.pi) % (2 * math.pi) - math.pi).abs().mean())   # 板の位相が親からどれだけ動いたか（ラジアン）
    print(name, out[name], f"{time.time() - t00:.0f} 秒", flush=True)
out["parent"] = evaluate(parent_p, TE); out["table_parent"] = evaluate(tg_par, TE); out["table_global"] = evaluate(tg_glb, TE)
inb = torch.tensor([tuple(k.split("の", 1)) in both for k in keys])                         # 両方の案に入る組だけの当たりも
if inb.any():
    out["both_diff"] = evaluate(prob(torch.cat([run(Pd[i:i + B].to(dev), len(VIEWS), NV + len(VIEWS)).cpu() for i in range(0, len(keys), B)]))[inb], TE[inb])
    out["both_parent"] = evaluate(parent_p[inb], TE[inb])
# 合成語ごとの「親より良くなった分」（ビット）: 例を出すため
Wd = torch.cat([run(Pd[i:i + B].to(dev), len(VIEWS), NV + len(VIEWS)).cpu() for i in range(0, len(keys), B)]); pd = prob(Wd)
gain = []
for i, kk in enumerate(keys):
    n = float(TE[i].sum())
    if n >= 20:
        gain.append((kk, pars[i], float(-(TE[i] * parent_p[i].log2()).sum() / n + (TE[i] * pd[i].log2()).sum() / n), n))
gain.sort(key=lambda t: -t[2])
res.update(n=len(keys), result=out, gain_top=gain[:20], gain_bottom=gain[-10:], sec=time.time() - t00, device=dev)
torch.save(dict(P=Pd, words=keys, parents=pars, views=VIEWS, pos=POS), f"{PRIV}/models/goban_compound_{a.set}.pt")
with open(a.out, "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in res.items() if k != "gain_bottom"}, ensure_ascii=False)[:3000])
