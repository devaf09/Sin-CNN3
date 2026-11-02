"""実験110: 穴埋めが当たるように構造を育てる（物差し: docs/task_yardstick.md）と、その構造の逆回し（並べ直し）

1. 育てる（解析の向き）:
   文の中の文節を 1 つ隠し（中身の語だけ隠す。尾の 助詞・助動詞・句読点 は見える）、残りの文節の中身の語から当てる。
   当て方 = 手がかりの語ごとの共鳴（一緒に出る強さ、正の PMI）を、重み付きで重ねる。重みは「隠した文節の尾・手がかりの文節の尾・左か右か・何文節離れているか・
   手がかりが文の最後か」の組ごとに 1 つの数 θ。θ を、穴埋めが当たるように（正しい語の見込みが上がるように）勾配で学ぶ。
   人の作った木（GiNZA の係り先）は学びに使わない。答えは材料の隠した語だけ。左右どちらにも係れる（日本語の「右だけ」を前提にしない）。
   比べ: 実験109（袋 0.044、並びの近さ 0.053、自分の木 0.053、GiNZA の木 0.075）。
   見るだけ: 学んだ重みが一番大きい手がかりが、GiNZA の木で隣（係り先か、係ってくる文節）かどうか。
2. 逆回し（生成の向き）: 同じ θ と共鳴で、文節の並びを当てる。文の文節を混ぜ、θ × 共鳴 の合計が一番大きい並び（探索、7 文節以下）を選ぶ。
   正しい並びとの一致（丸ごと・2 つずつの前後）を、でたらめ（2 つずつ 50%）と比べる。
材料: 実験107 と同じ（会話・青空文庫 1.5 万文、Web 小説は使わない）。文節の切り方は GiNZA のもの（切り方を学ぶのは次の段）。
結果: results/exp110.txt、docs/task_yardstick.md に追記、図 docs/figs/cloze_grown_110.svg。実行: python experiments/exp110_cloze_grown_structure.py
"""
import itertools, json, math, random, sys
from collections import Counter, defaultdict

import torch

src = open("experiments/exp109_structure_by_cloze.py", encoding="utf-8").read().split("m_self = Model()")[0]
exec(src)                                                                                      # 材料・中身の語・共鳴（PMI）・GiNZA の木の段数（実験109）
torch.manual_seed(0)
TAILS = [t for t, c in Counter(b["tail"] for s in tr for b in s["b"]).most_common(60)]; TI = {t: i for i, t in enumerate(TAILS)}
ti = lambda b: TI.get(b["tail"], len(TAILS))                                                    # よく出る 60 の尾 + その他
NT = len(TAILS) + 1


def key(s, a, c):
    """隠した文節 a と手がかり c の組の番号（尾 × 尾 × 向き・遠さ 8 通り × 文の最後か）。"""
    d = c - a; dd = (0 if d < 0 else 4) + min(abs(d), 4) - 1
    last = 1 if c == len(s["b"]) - 1 else 0
    return ((ti(s["b"][a]) * NT + ti(s["b"][c])) * 8 + dd) * 2 + last


NK = NT * NT * 8 * 2


def cases(data):
    out = []
    for s in data:
        for a, b in enumerate(s["b"]):
            if b["lem"] not in VI:
                continue
            ctx = [(key(s, a, c), VI[x["lem"]]) for c, x in enumerate(s["b"]) if c != a and x["lem"] in VI]
            if ctx:
                out.append((VI[b["lem"]], ctx, s, a))
    return out


TR, TE = cases(tr), cases(te)
print("教えの穴埋め", len(TR), "試し", len(TE), "組の種類", NK, flush=True)
theta = torch.zeros(NK, requires_grad=True)
opt = torch.optim.Adam([theta], lr=0.05)
MAXC = 24


def batch_logits(B):
    L = max(len(c) for _, c, _, _ in B)
    keys = torch.zeros(len(B), L, dtype=torch.long); ids = torch.zeros(len(B), L, dtype=torch.long); mask = torch.zeros(len(B), L, dtype=torch.bool)
    for i, (_, ctx, _, _) in enumerate(B):
        for j, (k, v) in enumerate(ctx):
            keys[i, j] = k; ids[i, j] = v; mask[i, j] = True
    w = theta[keys].masked_fill(~mask, -1e9).softmax(1)                                         # 手がかりの重み（学ぶ）
    sc = (w[:, :, None] * PMI[ids]).sum(1) + 0.1 * prior
    sc = sc.masked_fill(torch.zeros_like(sc, dtype=torch.bool).scatter_(1, ids.masked_fill(~mask, 0), mask), -1e9)   # 手がかりそのものは答えにしない
    return sc, w, keys, mask


def mrr(data):
    with torch.no_grad():
        tot = 0.0
        for i in range(0, len(data), 256):
            B = data[i:i + 256]; sc, _, _, _ = batch_logits(B)
            y = torch.tensor([t for t, _, _, _ in B])
            tot += (1.0 / ((sc > sc.gather(1, y[:, None])).sum(1) + 1).float()).sum().item()
    return tot / len(data)


res = dict(exp="110", n_train=len(TR), n_test=len(TE), curve=[])
print("学ぶ前（袋と同じ）", round(mrr(TE), 4), flush=True)
for ep in range(4):
    random.Random(ep).shuffle(TR)
    for i in range(0, len(TR), 256):
        B = TR[i:i + 256]; sc, _, _, _ = batch_logits(B)
        loss = torch.nn.functional.cross_entropy(sc, torch.tensor([t for t, _, _, _ in B]))
        opt.zero_grad(); loss.backward(); opt.step()
    v = mrr(TE); res["curve"].append(v); print("回", ep + 1, "穴埋め MRR", round(v, 4), flush=True)
# 見るだけ: 一番重い手がかりが GiNZA の木で隣か
hit = tot = 0; hit_near = 0
with torch.no_grad():
    for t, ctx, s, a in TE:
        n = len(s["b"]); cs = [c for c, x in enumerate(s["b"]) if c != a and x["lem"] in VI]
        ws = theta[torch.tensor([k for k, _ in ctx])]
        c_best = cs[int(ws.argmax())]
        tree_nb = {s["b"][a]["head"]} | {c for c, x in enumerate(s["b"]) if x["head"] == a}
        hit += c_best in tree_nb; tot += 1
        c_near = min(cs, key=lambda c: abs(c - a)); hit_near += c_near in tree_nb
res["top_is_tree_neighbor"] = hit / tot; res["nearest_is_tree_neighbor"] = hit_near / tot
print("一番重い手がかりが GiNZA の木で隣", round(hit / tot, 3), "／ 一番近い文節が木で隣", round(hit_near / tot, 3), flush=True)
# 学んだ重みの例: 隠した文節の尾ごとに、どの手がかり（尾・向き・遠さ）を重く見るか
DD = ["左1", "左2", "左3", "左4+", "右1", "右2", "右3", "右4+"]
ex = {}
with torch.no_grad():
    for t in ["が", "を", "に", "は", "の", "て", "た", "ます"]:
        if t not in TI:
            continue
        rows = []
        for tc in TAILS[:25]:
            for dd in range(8):
                for last in (0, 1):
                    k = ((TI[t] * NT + TI[tc]) * 8 + dd) * 2 + last
                    rows.append((float(theta[k]), f"{tc}・{DD[dd]}{'・文末' if last else ''}"))
        ex[t] = [(n, round(v, 2)) for v, n in sorted(rows, reverse=True)[:4]]
res["top_weights"] = ex; print(json.dumps(ex, ensure_ascii=False), flush=True)


# ---- 逆回し: 同じ θ と共鳴で、混ぜた文節の並びを当てる ----
def order_score(s, perm):
    """並び perm（元の文節の番号の列）の点 = 各文節を隠したとして、手がかりの重み × 共鳴（自分の語との）の合計。"""
    tot = 0.0; B = [s["b"][p] for p in perm]; s2 = dict(b=B)
    for a in range(len(B)):
        if B[a]["lem"] not in VI:
            continue
        ks, vs = [], []
        for c in range(len(B)):
            if c != a and B[c]["lem"] in VI:
                ks.append(key(s2, a, c)); vs.append(float(PMI[VI[B[c]["lem"]], VI[B[a]["lem"]]]))
        if ks:
            w = theta[torch.tensor(ks)]
            tot += float((torch.log_softmax(w, 0).exp() * torch.tensor(vs)).sum()) + 0.02 * float(w.logsumexp(0))
    return tot


ok = pair_ok = pair_tot = n_s = 0; rnd_ok = 0
with torch.no_grad():
    sel = [s for s in te if 3 <= len(s["b"]) <= 6][:300]
    for s in sel:
        n = len(s["b"]); perms = list(itertools.permutations(range(n)))
        best = max(perms, key=lambda p: order_score(s, p))
        ok += list(best) == list(range(n)); rnd_ok += 1 / len(perms); n_s += 1
        pos = {p: i for i, p in enumerate(best)}
        for x in range(n):
            for y in range(x + 1, n):
                pair_ok += pos[x] < pos[y]; pair_tot += 1
res["reorder"] = dict(n=n_s, exact=ok / n_s, exact_random=rnd_ok / n_s, pairwise=pair_ok / pair_tot)
print("逆回し（並べ直し）", res["reorder"], flush=True)
with open("results/exp110.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
# ---- 追加: 同じ試しの穴埋めで、決まった重み（並びの近さ・GiNZA の木）と比べる（実験109 と問の数が少し違うため）----
def mrr_fixed(kind):
    tot = 0.0
    with torch.no_grad():
        for t, ctx, s, a in TE:
            cs = [c for c, x in enumerate(s["b"]) if c != a and x["lem"] in VI]
            if kind == "並びの近さ":
                w = torch.tensor([0.5 ** (abs(c - a) - 1) for c in cs])
            else:
                D = tree_dist([b["head"] for b in s["b"]], len(s["b"])); w = torch.tensor([0.5 ** (D[a][c] - 1) if D[a][c] < 99 else 0.0 for c in cs])
            w = w / w.sum().clamp(min=1e-9); ids = [v for _, v in ctx]
            sc = (w[:, None] * PMI[ids]).sum(0) + 0.1 * prior; sc[ids] -= 1e9
            tot += 1.0 / (int((sc > sc[t]).sum()) + 1)
    return tot / len(TE)


res2 = dict(exp="110b", 並びの近さ=mrr_fixed("並びの近さ"), GiNZAの木=mrr_fixed("GiNZA の木"), 学んだ重み_最良=max(res["curve"]), 学ぶ前=None)
print(res2, flush=True)
with open("results/exp110.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res2, ensure_ascii=False) + "\n")
