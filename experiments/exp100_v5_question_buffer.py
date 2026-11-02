"""実験100: 語板 v5 を、質問の重みとバッファの変動に使う（持ち主の依頼 2026-11-02「質問に対し何に重みを追加するか、それとバッファ板の変動にも使いたい」）

1. 質問のどの語に重みを足すか: 実験97 は「動詞（名札で判定）を 2 倍」。ここでは v5 の品詞の観点の光から、語の「動詞らしさ」p を読み、重み = 1 + p にする（決まりなし）。
2. バッファの変動（答えの種類の予想）: 質問の述語の v5 の語板から、答えの役割の観点（何を → V:を、誰が → V:が、どこへ・何に → V:に、どこで → V:で、何と → V:と）の
   光を読み、「答えになりそうな名詞の仲間」（実験77 の 100 組）の割合を出す。答えの候補の節の名詞の仲間が、全体よりどれだけ多く予想されているか（持ち上がり）を
   点に掛ける: 点 × (1 + β × log 持ち上がり)。β = 0（予想なし）・0.3・0.6。
   例: 「太郎は何を買い取りましたか」→ 買い取る の V:を → 動物・品物の仲間 → 亀 の節が引き寄せられる。
比べ: 実験97（手の決まり + 動詞 2 倍）。浦島太郎の質問 20 + 使っていない 10。
結果: results/exp100.txt。実行: python experiments/exp100_v5_question_buffer.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import _h, VAL0

spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
T, S, DIM, own, cos, subtree_text, Q1, A1, Q2, A2, WH, CONTENT = (ns[k] for k in ("T", "S", "DIM", "own", "cos", "subtree_text", "Q1", "A1", "Q2", "A2", "WH", "CONTENT"))
sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"
NX = torch.load(f"{PRIV}/models/goban_next_web2.pt", weights_only=False); CL = {w: int(c) for w, c in zip(NX["words"], NX["cluster"])}
# ---- v5 の語板を読み、語ごとに 品詞の割合・動詞の観点の光 を出す ----
V5 = {}
for g in ("noun", "verb", "adj"):
    M = torch.load(f"{PRIV}/models/goban_v5_{g}.pt", weights_only=False)
    VIEWS, ALLV, NS = M["views"], M["all_views"], M["slot_ns"]; NOUT = NS + len(ALLV); N, K = M["P"].shape[1], M["P"].shape[2]
    H = _h("cpu", N); yy = torch.arange(N, dtype=torch.float32)
    WAVES = torch.stack([torch.exp(1j * 2 * math.pi * (ALLV.index(v) + 1) * yy / N) / math.sqrt(N) for v in VIEWS]).to(torch.complex64)
    need = {w[1] for q in Q1 + Q2 for b in q["bunsetsu"] for w in b["words"]} | {w for nodes in T for n in nodes for w in n["words"]}
    WI = {w: i for i, w in enumerate(M["words"])}; ids = [WI[w] for w in need if w in WI]
    if not ids:
        continue
    with torch.no_grad():
        u = torch.nn.functional.pad(WAVES, (0, N))[None].expand(len(ids), -1, -1); P = M["P"][ids]
        for k in range(K):
            u = torch.cat([u[..., :N] * torch.exp(1j * P[:, None, :, k]), u[..., N:]], -1)
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
        W = torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs()[..., VAL0:VAL0 + NOUT] ** 2
    for row, i in enumerate(ids):
        w = M["words"][i]; d = V5.setdefault(w, {})
        for k, v in enumerate(VIEWS):
            j = ALLV.index(v)
            if v == "品詞":
                p = torch.tensor([float(W[row, k, (r + 7 * j) % NS]) for r in range(14)]); d["pos"] = p / p.sum().clamp(min=1e-12)
            elif v.startswith("V:"):
                p = torch.tensor([float(W[row, k, (r + 7 * j) % NS]) for r in range(100)]); d[v] = p / p.sum().clamp(min=1e-12)
POS_TOP = M["pos_top"]; IV = POS_TOP.index("動詞")
cl_prior = torch.bincount(torch.tensor(list(CL.values())), minlength=100).float(); cl_prior /= cl_prior.sum()
verbness = lambda w: float(V5[w]["pos"][IV]) if w in V5 and "pos" in V5[w] else None
TO_VIEW = {"を": "V:を", "が": "V:が", "は": "V:が", "に": "V:に", "へ": "V:に", "で": "V:で", "と": "V:と", "引用": "V:と"}


def question(q, use_pos):
    txt = q["sentence"]
    wh = next((k for k in ("名前", "何を", "何に", "何が", "何と", "どこへ", "どこで", "誰", "どんな", "何") if k in txt), "何")
    r = torch.zeros(DIM)
    for x in WH[wh]:
        if x in ns["RI"]:
            r[len(ns["VOC"]) + ns["RI"][x]] = 1.0
    stop = {"何", "誰", "どこ", "名前", "どんな", "最後"}
    b = torch.zeros(DIM); verbs = []
    for bb in q["bunsetsu"]:
        for w in bb["words"]:
            if not w[2].startswith(CONTENT) or w[1] in stop or w[0] in stop:
                continue
            is_verb_tag = w[2].startswith("動詞") or (w[2].startswith("名詞") and any(x[1] == "為る" for x in bb["words"]))
            vb = verbness(w[1]) if use_pos else None
            wt = (1.0 + vb) if vb is not None else (2.0 if is_verb_tag else 1.0)            # v5 の品詞の光で重み（無い語は名札の判定）
            b += wt * own([w[1]], None)
            if is_verb_tag and w[1] not in ("為る", "居る", "成る"):
                verbs.append(w[1])
    return b, r, wh, verbs


def expect(wh, verbs):
    """答えの役割の観点で、質問の述語の語板から「答えになりそうな名詞の仲間」の割合。"""
    views = {TO_VIEW[x] for x in WH[wh] if x in TO_VIEW}
    ps = [V5[v][vw] for v in verbs if v in V5 for vw in views if vw in V5[v]]
    return (sum(ps) / len(ps)) if ps else None


def answer(q, beta, use_pos):
    b, r, wh, verbs = question(q, use_pos); ex = expect(wh, verbs) if beta > 0 else None
    banned = set()
    for it in range(3):
        si = max((i for i in range(len(T)) if i not in banned), key=lambda i: cos(next(m for m in T[i] if m["head"] == -1)["G"], b))
        best = (-1, None)
        for n in T[si]:
            rr = float(n["g"][len(ns["VOC"]):len(ns["VOC"]) + len(ns["ROLES"])] @ r[len(ns["VOC"]):len(ns["VOC"]) + len(ns["ROLES"])])
            if rr <= 0:
                continue
            sc = rr * cos(n["C"], b)
            if ex is not None:
                cls = [CL[w] for w in n["words"] if w in CL]
                if cls:
                    lift = max(float(ex[c] / cl_prior[c]) for c in cls)
                    sc *= max(0.05, 1 + beta * math.log(max(lift, 1e-3)))                  # 答えの種類の予想で引き寄せる（バッファの変動）
            if sc > best[0]:
                best = (sc, n)
        if best[1] is not None and best[0] >= 0.3:
            return subtree_text(T[si], best[1]["id"])
        banned.add(si)
    return subtree_text(T[si], best[1]["id"]) if best[1] is not None else "（なし）"


res = dict(exp="100")
for name, beta, use_pos in (("実験97（名札で動詞 2 倍）", 0.0, False), ("v5 の品詞の光で重み", 0.0, True), ("+ 答えの種類の予想 β=0.3", 0.3, True), ("+ 答えの種類の予想 β=0.6", 0.6, True)):
    ok = [0, 0]; wrong = []
    for k, (Q, A) in enumerate(((Q1, A1), (Q2, A2))):
        for q, ans in zip(Q, A):
            a = answer(q, beta, use_pos); good = any(x in a for x in ans) or (a != "（なし）" and any(a in x for x in ans))
            ok[k] += good
            if not good:
                wrong.append(q["sentence"])
    res[name] = dict(q20=ok[0], q10=ok[1], wrong=wrong); print(name, ok, wrong, flush=True)
# 例: 動詞の語板が予想する答えの仲間（上位 3 組の代表の名詞）
ex_rows = {}
members = {}
for w, c in CL.items():
    members.setdefault(c, []).append(w)
for v, vw in (("買い取る", "V:を"), ("渡す", "V:を"), ("開ける", "V:を"), ("歓迎", "V:が"), ("乗る", "V:に"), ("到着", "V:に")):
    if v in V5 and vw in V5[v]:
        lift = V5[v][vw] / cl_prior
        top = torch.argsort(lift, descending=True)[:3]
        ex_rows[f"{v} {vw}"] = [("・".join(members[int(c)][:6]), round(float(lift[c]), 1)) for c in top]
res["examples"] = ex_rows
print(json.dumps(ex_rows, ensure_ascii=False)[:1500])
with open("results/exp100.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
