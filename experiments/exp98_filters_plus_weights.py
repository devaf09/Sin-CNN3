"""実験98: 学んだフィルター（実験95）とはい・いいえ（実験94）に、質問側の重み（述語 2 倍、実験97）を足して、同じ質問をもう一度（持ち主の依頼 2026-11-02）

実験95 の「両方」（疑問フィルター + 役割フィルター）と、実験94（否定の向き）の答え探しで、バッファを作る所だけを実験97 の「動詞を重く」に置き換える。
結果: results/exp98.txt。実行: python experiments/exp98_filters_plus_weights.py
"""
import json, os, sys

import torch

sys.argv = [sys.argv[0]]; sys.stdout.reconfigure(encoding="utf-8")
here = os.path.dirname(__file__)


def load(name, cut):
    p = os.path.join(here, name); src = open(p, encoding="utf-8").read()
    if cut:
        src = src.split(cut)[0]
    ns = {"__file__": p}; exec(compile(src, p, "exec"), ns); return ns


# 実験97 の「動詞を重く」のバッファの作り方
n97 = load("exp97_buffer_weights.py", "res = dict(exp=\"97\")")
res = dict(exp="98")
# --- 実験95（フィルター）に足す ---
n95 = load("exp95_filters_in_tree.py", "res = dict(exp=\"95\")")
n95_base = dict(q20=0, q10=0)
for label, use_w in (("フィルター", False), ("フィルター + 述語 2 倍", True)):
    n95["question"] = n97["make_question"]("動詞を重く") if use_w else n95["ns"]["question"]
    ok = [0, 0]
    for k, (Q, A) in enumerate(((n95["Q1"], n95["A1"]), (n95["Q2"], n95["A2"]))):
        for q, ans in zip(Q, A):
            a = n95["answer"](q, "両方")
            ok[k] += any(x in a for x in ans) or (a != "（なし）" and any(a in x for x in ans))
    res[label] = dict(q20=ok[0], q10=ok[1]); print(label, "20 問", ok[0], "未使用 10 問", ok[1], flush=True)
# --- 実験94（はい・いいえ）に足す ---
p94 = os.path.join(here, "exp94_negation_buffer.py"); src94 = open(p94, encoding="utf-8").read()
for label, use_w in (("はい・いいえ", False), ("はい・いいえ + 述語 2 倍", True)):
    s = src94
    if use_w:                                                                                # バッファを作る所を「述語 2 倍」に（質問の述語の名札を 2 倍）
        s = s.replace("        b += own(ws, None)", "        for w_ in ws:\n            vb_ = any(x[1] == w_ and (x[2].startswith('動詞') or (x[2].startswith('名詞') and any(y[1] == '為る' for y in bb['words']))) for x in bb['words'])\n            b += (2.0 if vb_ else 1.0) * own([w_], None)")
    s = s.split("res = dict(exp=\"94\"")[0]
    ns = {"__file__": p94}; exec(compile(s, p94, "exec"), ns)
    c = sum(r["ok"] for r in ns["rows"]); res[label] = dict(correct=c, n=len(ns["rows"]), wrong=[r["q"] for r in ns["rows"] if not r["ok"]])
    print(label, c, "/", len(ns["rows"]), res[label]["wrong"], flush=True)
with open("results/exp98.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
