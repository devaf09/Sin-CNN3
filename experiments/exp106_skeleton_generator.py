"""実験106: 骨格生成器のミニ実験（別の AI の助言 2026-11-02 の提案 2「動詞の格要求から骨格のスロット列を自動で決める」、持ち主「うん」）

骨格 = 述語 1 つ + その述語に係る文節の格（が・は・を・に・で・と・へ・から・も・まで・より）の組。
問い: 述語（と、文の中での係り方 ROOT・advcl・acl…）だけから、どの格のスロットを立てるかを当てられるか。
決め方:
  A 事前: 品詞と係り方ごとに、いちばん多い格の組。
  B 述語の要求（数え上げ）: その述語で格 c が出る割合 P(c | 述語) が しきい値 θ を超えたら c のスロットを立てる。数が 5 未満なら A。
  C B + 係り方: P(c | 述語, 係り方)。数が少なければ B、さらに A へ戻る。
  D 語板 v5 の光で借りる: 教えに少ない（5 未満）述語は、語板 v5 の動詞の光（V:が・V:を… の観点の出口の明るさ）が近い 10 語の要求の平均を使う。
評価: 試しの述語ごとに、立てたスロットの組と実際の格の組を比べる（組が丸ごと一致・格ごとの F1）。
      「実際の組」には省略（主語を言わない等）が入るので、上限は 100% にならない。上限の目安 = 試しの組を、同じ述語・係り方のいちばん多い組で当てた割合（試しで数えた天井）。
材料: Sin-CNN3_private/data/views/predicate_frames.json（会話と青空文庫 4 万文、GiNZA。Web 小説は使わない）。
結果: results/exp106.txt、docs/skeleton_generator_106.md、図 docs/figs/skeleton_generator_106.svg。実行: python experiments/exp106_skeleton_generator.py
"""
import json, math, os, sys
from collections import Counter, defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import _h, VAL0

sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"
CASES = ["が", "は", "を", "に", "で", "と", "へ", "から", "も", "まで", "より"]
F = json.load(open(f"{PRIV}/data/views/predicate_frames.json", encoding="utf-8"))
tr = [f for f in F if f["split"] == "train"]; te = [f for f in F if f["split"] == "test"]
S = lambda f: frozenset(f["slots"])
print("述語 教え", len(tr), "試し", len(te), flush=True)
# ---- 数え上げ ----
n_lem = Counter(f["lemma"] for f in tr); n_ld = Counter((f["lemma"], f["dep"]) for f in tr)
c_lem = defaultdict(Counter); c_ld = defaultdict(Counter); sets_pd = defaultdict(Counter); sets_ld_te = defaultdict(Counter)
for f in tr:
    for c in S(f):
        c_lem[f["lemma"]][c] += 1; c_ld[(f["lemma"], f["dep"])][c] += 1
    sets_pd[(f["pos"], f["dep"])][S(f)] += 1
for f in te:
    sets_ld_te[(f["lemma"], f["dep"])][S(f)] += 1
prior = {k: v.most_common(1)[0][0] for k, v in sets_pd.items()}
# ---- 語板 v5 の動詞の光（V: の観点の出口の明るさを並べたもの）----
M = torch.load(f"{PRIV}/models/goban_v5_verb.pt", weights_only=False)
VIEWS, ALLV, NS = M["views"], M["all_views"], M["slot_ns"]; NOUT = NS + len(ALLV); N, K = M["P"].shape[1], M["P"].shape[2]
VV = [v for v in VIEWS if v.startswith("V:")]
H = _h("cpu", N); yy = torch.arange(N, dtype=torch.float32)
WAVES = torch.stack([torch.exp(1j * 2 * math.pi * (ALLV.index(v) + 1) * yy / N) / math.sqrt(N) for v in VV]).to(torch.complex64)
LIGHT = {}
with torch.no_grad():
    for s0 in range(0, len(M["words"]), 500):
        P = M["P"][s0:s0 + 500]; u = torch.nn.functional.pad(WAVES, (0, N))[None].expand(len(P), -1, -1)
        for k in range(K):
            u = torch.cat([u[..., :N] * torch.exp(1j * P[:, None, :, k]), u[..., N:]], -1)
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
        out = torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs()[..., VAL0:VAL0 + NOUT] ** 2       # (語, 観点, 出口)
        for i, w in enumerate(M["words"][s0:s0 + 500]):
            LIGHT[w] = (out[i] / out[i].sum(-1, keepdim=True).clamp(min=1e-12)).flatten()
LW = [w for w in LIGHT if n_lem[w] >= 5]; LM = torch.stack([LIGHT[w] for w in LW]); LM = LM / LM.norm(dim=1, keepdim=True)
print("v5 の動詞の光", len(LIGHT), "語、借りる先の候補", len(LW), flush=True)


def prob_lemma(lem):
    return {c: c_lem[lem][c] / n_lem[lem] for c in CASES}


def borrow(lem, k=10):
    """語板 v5 の光が近い 10 語の要求の平均（自分は除く）。"""
    if lem not in LIGHT:
        return None
    v = LIGHT[lem] / LIGHT[lem].norm(); sim = LM @ v
    idx = [i for i in torch.argsort(sim, descending=True).tolist() if LW[i] != lem][:k]
    ps = [prob_lemma(LW[i]) for i in idx]
    return {c: sum(p[c] for p in ps) / len(ps) for c in CASES}, [LW[i] for i in idx[:5]]


def predict(f, how, th):
    lem, dep = f["lemma"], f["dep"]
    if how == "A":
        return prior.get((f["pos"], dep), frozenset())
    p = None
    if how == "C" and n_ld[(lem, dep)] >= 5:
        p = {c: c_ld[(lem, dep)][c] / n_ld[(lem, dep)] for c in CASES}
    if p is None and n_lem[lem] >= 5:
        p = prob_lemma(lem)
    if p is None and how == "D":
        b = borrow(lem); p = b[0] if b else None
    if p is None:
        return prior.get((f["pos"], dep), frozenset())
    return frozenset(c for c in CASES if p[c] > th)


def score(pairs):
    ex = sum(a == b for a, b in pairs) / len(pairs)
    tp = sum(len(a & b) for a, b in pairs); fp = sum(len(a - b) for a, b in pairs); fn = sum(len(b - a) for a, b in pairs)
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(exact=ex, f1=2 * pr * rc / max(pr + rc, 1e-9), prec=pr, rec=rc)


res = dict(exp="106", n_train=len(tr), n_test=len(te), runs=[])
rare = [f for f in te if n_lem[f["lemma"]] < 5 and f["lemma"] in LIGHT]
ceil = sum(v.most_common(1)[0][1] for v in sets_ld_te.values()) / len(te)
print("試しで数えた天井（同じ述語・係り方のいちばん多い組）", round(ceil, 3), "教えに少ない述語（v5 にはある）", len(rare), flush=True)
for th in (0.2, 0.3, 0.4, 0.5):
    for how in "ABCD":
        a = score([(predict(f, how, th), S(f)) for f in te]); r = score([(predict(f, how, th), S(f)) for f in rare]) if rare else {}
        res["runs"].append(dict(how=how, th=th, all=a, rare=r))
        print("θ", th, how, "全体 一致", round(a["exact"], 3), "F1", round(a["f1"], 3), "／ 少ない述語 一致", round(r.get("exact", 0), 3), "F1", round(r.get("f1", 0), 3), flush=True)
res["ceiling"] = ceil; res["n_rare"] = len(rare)
# 例: よく出る述語の要求（格の割合）と、v5 で借りた例
EXV = ["食べる", "行く", "見る", "言う", "乗る", "渡す", "住む", "思う", "会う", "似る", "帰る", "開ける"]
res["frames"] = {v: {c: round(c_lem[v][c] / n_lem[v], 2) for c in CASES if n_lem[v] and c_lem[v][c] / n_lem[v] >= 0.05} for v in EXV if n_lem[v]}
res["borrow_examples"] = {}
for f in rare[:400]:
    b = borrow(f["lemma"])
    if b and len(res["borrow_examples"]) < 12:
        res["borrow_examples"][f["lemma"]] = dict(neighbors=b[1], frame={c: round(x, 2) for c, x in b[0].items() if x >= 0.1})
res["set_dist"] = [(sorted(k), v) for k, v in Counter(S(f) for f in te).most_common(10)]
print(json.dumps(res["frames"], ensure_ascii=False)); print(json.dumps(res["borrow_examples"], ensure_ascii=False)[:1500]); print(res["set_dist"])
with open("results/exp106.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
# ---- 追加: 格が 1 つ以上出た述語だけで、「いちばん要求の強い格」が実際に出た格に入るか（省略の多さを外した見方）----
def top_case(f, how):
    lem, dep = f["lemma"], f["dep"]; p = None
    if how == "A":
        cnt = Counter()
        for s_, n_ in sets_pd[(f["pos"], dep)].items():
            for c in s_:
                cnt[c] += n_
        return cnt.most_common(1)[0][0] if cnt else "は"
    if how == "C" and n_ld[(lem, dep)] >= 5:
        p = {c: c_ld[(lem, dep)][c] / n_ld[(lem, dep)] for c in CASES}
    if p is None and n_lem[lem] >= 5:
        p = prob_lemma(lem)
    if p is None and how == "D":
        b = borrow(lem); p = b[0] if b else None
    if p is None:
        return top_case(f, "A")
    return max(CASES, key=lambda c: p[c])


withslot = [f for f in te if S(f)]; rare_ws = [f for f in rare if S(f)]
res2 = {}
for how in "ABCD":
    a = sum(top_case(f, how) in S(f) for f in withslot) / len(withslot); r = sum(top_case(f, how) in S(f) for f in rare_ws) / max(len(rare_ws), 1)
    res2[how] = dict(all=a, rare=r); print("格が出た述語", len(withslot), how, "いちばん強い格が当たる", round(a, 3), "／ 少ない述語", len(rare_ws), round(r, 3), flush=True)
with open("results/exp106.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(dict(exp="106b", n_withslot=len(withslot), n_rare_withslot=len(rare_ws), top_case=res2, empty_share=sum(not S(f) for f in te) / len(te)), ensure_ascii=False) + "\n")
