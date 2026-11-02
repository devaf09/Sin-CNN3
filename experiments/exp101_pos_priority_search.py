"""実験101: 品詞によって検索の優劣をつける（持ち主の案 2026-11-02「属性重みを比較する際、品詞によって検索の優劣をつけるのはどうだろうか」）

重みを混ぜて 1 つの点にする（実験89・97）のではなく、品詞の順に段階で絞る:
  ① 述語（動詞・名詞 + する）の手がかりの光だけで、幹（文）を点数順に並べ、上位（1 位との差が δ 以内）を残す。
  ② 残った幹の中で、名詞の手がかりの光で幹を選び直す。
  ③ 選んだ幹の中で、節を選ぶ（まわりの光 × 役割の合い、実験89 と同じ）。
  述語の手がかりが無い質問は ② から。δ = 0.05・0.1・0.2。
比べ: 実験97（述語 2 倍で 1 つの点）29/30、実験89（重みなし）27/30。浦島太郎の質問 20 + 使っていない 10。
結果: results/exp101.txt。実行: python experiments/exp101_pos_priority_search.py
"""
import json, os, sys

import torch

spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
T, DIM, own, cos, subtree_text, Q1, A1, Q2, A2, CONTENT, VOC, ROLES = (ns[k] for k in ("T", "DIM", "own", "cos", "subtree_text", "Q1", "A1", "Q2", "A2", "CONTENT", "VOC", "ROLES"))
sys.stdout.reconfigure(encoding="utf-8")
orig = ns["question"]


def split_buffer(q):
    """質問の手がかりを、述語の光 bv と 名詞などの光 bn に分ける。"""
    stop = {"何", "誰", "どこ", "名前", "どんな", "最後"}
    bv = torch.zeros(DIM); bn = torch.zeros(DIM)
    for bb in q["bunsetsu"]:
        for w in bb["words"]:
            if not w[2].startswith(CONTENT) or w[1] in stop or w[0] in stop:
                continue
            is_verb = w[2].startswith("動詞") or (w[2].startswith("名詞") and any(x[1] == "為る" for x in bb["words"]))
            if is_verb and w[1] in ("為る", "居る", "成る"):
                continue
            (bv if is_verb else bn).__iadd__(own([w[1]], None))
    return bv, bn


def answer(q, delta):
    _, r = orig(q); bv, bn = split_buffer(q); b = bv + bn
    roots = [next(m for m in nodes if m["head"] == -1) for nodes in T]
    cand = list(range(len(T)))
    if float(bv.norm()) > 0:                                                                 # ① 述語で幹を絞る
        sv = [cos(rt["G"], bv) for rt in roots]; top = max(sv)
        cand = [i for i in cand if sv[i] >= top - delta]
    si = max(cand, key=lambda i: cos(roots[i]["G"], bn) if float(bn.norm()) > 0 else 0)       # ② 名詞で選び直す
    best = (-1, None)
    for n in T[si]:                                                                          # ③ 節を選ぶ
        rr = float(n["g"][len(VOC):len(VOC) + len(ROLES)] @ r[len(VOC):len(VOC) + len(ROLES)])
        if rr > 0 and rr * cos(n["C"], b) > best[0]:
            best = (rr * cos(n["C"], b), n)
    return subtree_text(T[si], best[1]["id"]) if best[1] is not None else "（なし）"


res = dict(exp="101")
for delta in (0.05, 0.1, 0.2):
    ok = [0, 0]; wrong = []
    for k, (Q, A) in enumerate(((Q1, A1), (Q2, A2))):
        for q, ans in zip(Q, A):
            a = answer(q, delta); good = any(x in a for x in ans) or (a != "（なし）" and any(a in x for x in ans)); ok[k] += good
            if not good:
                wrong.append(f"{q['sentence']} → {a}")
    res[f"δ={delta}"] = dict(q20=ok[0], q10=ok[1], wrong=wrong); print("δ", delta, ok, wrong, flush=True)
with open("results/exp101.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
