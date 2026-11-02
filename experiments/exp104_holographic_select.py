"""実験104: ホログラフィックな物語の光（実験103）で、質問がどの枠かを選ぶところに「まわり」と「掛け算」を入れる
（持ち主 2026-11-02「バッファと品詞の重み付けで同じ様に掛け算と離れるほど重みを下げるやり方はどうかな」）

実験103: 覚えること 98%・次の出来事 19/19 だが、質問 12/30。外れは「質問がどの枠か」の選び方。
ここでは、手がかりの語ごとに、枠ごとの重なり m[枠, 語] を取り出す（物語 ⊙ 鍵^-1 と 手がかりの語の光の重なり。全部の枠を同時に計算できる）。
  まわり: M[k, 語] = Σ_枠m γ^距離(k, m) × m[m, 語]。距離 = 同じ文の木の上で枠の中心どうしが何段離れているか（ほかの文は 文の差 × 3 段）。γ = 0.5。
          （文章樹の「まわりの光」と同じ考え。離れた枠ほど薄く足す）
  点の付け方: 足し算 = Σ 重み × M、掛け算 = Π (ε + max(M, 0))^重み（どの手がかりも揃っている枠が勝つ = AND）。重みは 述語 2・ほか 1（実験97）。ε = 0.01。
比べ: 4 通り（まわり なし/あり × 足し算/掛け算）。D = 4096・16384。
結果: results/exp104.txt、docs/holographic_select_104.md、図 docs/figs/holographic_select_104.svg。実行: python experiments/exp104_holographic_select.py
"""
import json, math, os, sys

import torch

spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
T, S, V4, Q1, A1, Q2, A2, WH, CONTENT = (ns[k] for k in ("T", "S", "V4", "Q1", "A1", "Q2", "A2", "WH", "CONTENT"))
same = ns["build"].__globals__["same"]; subtree_text = ns["subtree_text"]
sys.stdout.reconfigure(encoding="utf-8")
ROLE = {"が": "主語", "は": "主語", "も": "主語", "を": "目的語", "に": "行き先", "へ": "行き先", "で": "場所", "から": "起点", "と": "相手", "という": "名前",
        "どんな": "様子", "の": "持ち主", "引用": "言った中身"}
ROLES = sorted(set(ROLE.values())) + ["述語", "ほか", "係り先", "袋"]
WH_ROLE = {"何を": ["目的語", "ほか"], "何に": ["行き先"], "何が": ["主語"], "何と": ["言った中身"], "どこへ": ["行き先"], "どこで": ["場所"], "誰": ["主語", "行き先"],
           "どんな": ["様子"], "名前": ["名前"], "何": ["目的語", "主語", "行き先"]}


def main_word(n):
    ws = [w for w in n["words"] if w not in ("為る", "居る", "成る", "おる", "様")]
    return ws[-1] if n["role"] in ("どんな",) and ws else (ws[0] if ws else None)


def word_of(n):
    """枠に入れる語: 名前の節は語をつなげる（浦島+太郎）、ほかは主な語。"""
    if n["role"] == "という" and n["words"]:
        return "".join(w for w in n["words"] if w not in ("言う", "と"))
    return main_word(n)


GAMMA, EPS = 0.5, 0.01
NEAR, COMB, ANS_NEAR = False, "足し算", False
BAG = 1.0                                                                                      # 袋（役割なしの語の束）の重み


def run(D, mode, seed=0):
    g = torch.Generator().manual_seed(seed)
    rand_phase = lambda: torch.exp(1j * 2 * math.pi * torch.rand(D, generator=g))
    vocab = sorted({w for nodes in T for n in nodes for w in n["words"]} | {word_of(n) for nodes in T for n in nodes if word_of(n)} | {w[1] for q in Q1 + Q2 for b in q["bunsetsu"] for w in b["words"]})
    proj = torch.randn(next(iter(V4.values())).shape[0], D, generator=g) / 30                 # 語板 v4 の光 → 位相（意味の近い語は位相が少し似る）
    W = {}
    for w in vocab:
        ph = 2 * math.pi * torch.rand(D, generator=g)
        if w in V4:
            ph = ph + 0.3 * (V4[w] @ proj) * 2 * math.pi
        W[w] = torch.exp(1j * ph)
    R = {r: rand_phase() for r in ROLES}
    step = torch.exp(1j * 2 * math.pi * torch.rand(D, generator=g))                           # 時刻の 1 歩（位相を回す量）
    tau = lambda k: step ** k
    sub = torch.exp(1j * 2 * math.pi * torch.rand(D, generator=g))                            # 文の中の 1 歩
    # 出来事の作り方:
    #   "文": 1 文 = 1 出来事（根の述語 + 根に係る節）。1 回目。
    #   "枠": 子を持つ節ごとに 1 つの枠（中心 = その節の語、まわり = 子の節を役割で結ぶ、親の節の語 = 「係り先」）。1 文に複数の枠。
    #         どの枠にも「袋」（役割なしの語の束 = 袋 ⊙ 語）を足す。質問の時刻選びは 役割つき + 袋 の両方で重ねる。
    events = []; centers = []
    for si, nodes in enumerate(T):
        if mode == "文":
            root = next(n for n in nodes if n["head"] == -1); e = torch.zeros(D, dtype=torch.complex64); items = []
            pw = main_word(root)
            if pw:
                e += R["述語"] * W[pw]; items.append(("述語", pw, root["text"]))
            for n in nodes:
                if n is root:
                    continue
                mw = main_word(n)
                if not mw:
                    continue
                role = ROLE.get(n["role"], "ほか")
                if n["head"] == root["id"] or role in ("様子", "名前", "持ち主", "言った中身"):
                    e += R[role] * W[mw]; items.append((role, mw, n["text"]))
            events.append((e, items, si)); centers.append((si, root["id"]))
            continue
        byid = {n["id"]: n for n in nodes}
        for c in nodes:
            kids = [n for n in nodes if n["head"] == c["id"] and word_of(n)]
            if not kids or not word_of(c):
                continue
            e = torch.zeros(D, dtype=torch.complex64); items = [("述語", word_of(c), c["text"])]
            if c["head"] in byid and word_of(byid[c["head"]]):
                items.append(("係り先", word_of(byid[c["head"]]), byid[c["head"]]["text"]))
            items += [(ROLE.get(n["role"], "ほか"), word_of(n), n["text"]) for n in kids]
            for r, w, _ in items:
                e += R[r] * W[w] + BAG * R["袋"] * W[w]
            events.append((e, items, si)); centers.append((si, c["id"]))
    # 時刻の鍵: 文の番号 k → τ^k、文の中の枠の番号 j → σ^j（文の中でも順番がある）
    keys, cnt = [], {}
    for _, _, si in events:
        j = cnt.get(si, 0); cnt[si] = j + 1; keys.append(tau(si) * sub ** j)
    story = sum(keys[k] * e for k, (e, _, _) in enumerate(events))                               # 1 枚の光に 20 文
    def tree_dist(si, a, b):
        heads = {n["id"]: n["head"] for n in T[si]}
        up = lambda x: [x] + (up(heads[x]) if heads[x] != -1 else [])
        pa, pb = up(a), up(b); common = next(x for x in pa if x in pb)
        return pa.index(common) + pb.index(common)
    NE = len(events)
    dist = torch.tensor([[float(tree_dist(centers[a][0], centers[a][1], centers[b][1])) if centers[a][0] == centers[b][0] else 3.0 * abs(centers[a][0] - centers[b][0])
                          for b in range(NE)] for a in range(NE)])
    unit = lambda v: v / v.abs().pow(2).sum().sqrt().clamp(min=1e-9)
    sim = lambda a, b: float((unit(a).conj() * unit(b)).sum().real)

    def cleanup(v, cands):
        return max(cands, key=lambda w: sim(v, W[w]))

    def ask(q):
        txt = q["sentence"]; wh = next((k for k in ("名前", "何を", "何に", "何が", "何と", "どこへ", "どこで", "誰", "どんな", "何") if k in txt), "何")
        stop = {"何", "誰", "どこ", "名前", "どんな", "最後"}; probe = torch.zeros(D, dtype=torch.complex64); cues = []
        for b in q["bunsetsu"]:
            for w in b["words"]:
                if w[2].startswith(CONTENT) and w[1] not in stop and w[0] not in stop and w[1] in W:
                    v = w[2].startswith("動詞") or (w[2].startswith("名詞") and any(x[1] == "為る" for x in b["words"]))
                    if v and w[1] in ("為る", "居る", "成る"):
                        continue
                    role = "述語" if v else ROLE.get(ns["build"].__globals__["role"](dict(words=[x[:3] for x in b["words"]], text=b["text"], dep="x")), "ほか")
                    probe += (2.0 if v else 1.0) * (R[role] + BAG * R["袋"]) * W[w[1]]; cues.append((2.0 if v else 1.0, (R[role] + BAG * R["袋"]) * W[w[1]]))                         # 手がかり: 役割 ⊙ 語（述語 2 倍、実験97）
        if not cues:
            return "（なし）", 0
        unb = [story * keys[j].conj() for j in range(NE)]                                      # 枠ごとに取り出した光（並列にできる）
        m = torch.tensor([[sim(u, cv) for _, cv in cues] for u in unb])                         # m[枠, 手がかりの語]
        wts = torch.tensor([wt for wt, _ in cues])
        M = (GAMMA ** dist) @ m if NEAR else m                                                 # まわり: 離れた枠ほど薄く足す
        score = (M * wts).sum(1) if COMB == "足し算" else ((EPS + M.clamp(min=0)).log() * wts).sum(1)   # 掛け算は log の足し算で
        k = int(score.argmax())
        qwords = {w[1] for b in q["bunsetsu"] for w in b["words"]}
        frames = [k] if not ANS_NEAR else [j for j in range(NE) if dist[k, j] <= 2]             # 答えを探す枠: 選んだ枠だけ / まわりも（距離 2 まで）
        best = (-1e9, None, None)
        for j in frames:
            text = {it[1]: it[2] for it in events[j][1] if len(it) > 2}
            cands = sorted({it[1] for it in events[j][1]} - qwords)
            for ri, r in enumerate(WH_ROLE[wh] + ["係り先"]):                                   # 役割の順（目的語 → ほか → 係り先）は 0.8 倍ずつ
                for w in cands:
                    sc = sim(unb[j] * R[r].conj(), W[w])
                    if sc <= 0.025:
                        continue
                    sc *= (0.8 ** ri) * (GAMMA ** float(dist[k, j]))                          # 離れた枠ほど薄く
                    if sc > best[0]:
                        best = (sc, text.get(w, w), j)
        if best[1] is None:
            return "（なし）", events[k][2]
        return best[1], events[best[2]][2]

    out = []
    for Q, A in ((Q1, A1), (Q2, A2)):
        for q, ans in zip(Q, A):
            a, k = ask(q); ok = any(a in x or x in a for x in ans) if a != "（なし）" else False
            out.append(dict(q=q["sentence"], ans=a, sent=k + 1, ok=ok))
    # 次の出来事: 時刻 k の次（τ^{k+1}）の述語を取り出す
    nxt = []
    first = {}
    for k, (_, _, si) in enumerate(events):
        first.setdefault(si, k)
    verbs = sorted({it[1] for e in events for it in e[1] if it[0] == "述語"})
    for si in sorted(first)[:-1]:
        if si + 1 not in first:
            continue
        ev = story * tau(si + 1).conj() * R["述語"].conj()                                    # 「次」 = 位相を 1 歩回すだけ
        got = cleanup(ev, verbs); want = events[first[si + 1]][1][0][1]
        nxt.append(got == want)
    # 取り出しの雑音: 正しい語との重なりと、ほかの語との重なり
    sig, noi, acc = [], [], []
    for k, (e, items, _) in enumerate(events):
        ev = story * keys[k].conj()
        for it in items:
            v = ev * R[it[0]].conj(); sig.append(sim(v, W[it[1]]))
            noi.append(max(sim(v, W[w]) for w in list(W)[:200] if w != it[1]))
            acc.append(cleanup(v, list(W)) == it[1])                                           # 時刻と役割が分かっているとき、全語（約 500）から正しく取り出せるか
    return dict(D=D, mode=mode, n_events=len(events), q20=sum(r["ok"] for r in out[:20]), q10=sum(r["ok"] for r in out[20:]), next_ok=sum(nxt), next_n=len(nxt),
                signal=sum(sig) / len(sig), recall=sum(acc) / len(acc), n_items=len(acc), noise=sum(noi) / len(noi), rows=out, events=[[si + 1] + [list(x[:2]) for x in it] for _, it, si in events])


runs = []
for D in (4096, 16384):
    for near in (False, True):
        for comb in ("足し算", "掛け算"):
            for ans_near in (False, True):
                NEAR, COMB, ANS_NEAR = near, comb, ans_near
                r = run(D, "枠"); r.update(near=near, comb=comb, ans_near=ans_near); runs.append(r)
                print("D", D, "選ぶ:", "まわり" if near else "まわりなし", comb, "答え:", "まわりも" if ans_near else "選んだ枠", "質問", r["q20"], "+", r["q10"], flush=True)
res = dict(exp="104", gamma=GAMMA, eps=EPS, runs=runs)
with open("results/exp104.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
