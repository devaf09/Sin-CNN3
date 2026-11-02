"""実験102: 品詞の優劣の強さを、語のありふれ具合で変える（実験101 の続き）

質問の手がかりの語ごとの重み（バッファの側）:
  df(w) = その語が出る文の数（文章の名札から数える、20 文）、珍しさ = log(20 / df)。
  A 述語 2 倍（実験97）／ B 述語だけ 1 + 珍しさ ／ C 全部の語 1 + 珍しさ ／ D 述語 1 + 2 × 珍しさ、名詞 1 + 珍しさ。
浦島太郎 30 問。結果: results/exp102.txt。実行: python experiments/exp102_rarity_weights.py
"""
import json, math, os, sys

import torch

spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
T, DIM, own, run_all, Q1, A1, Q2, A2, CONTENT = (ns[k] for k in ("T", "DIM", "own", "run_all", "Q1", "A1", "Q2", "A2", "CONTENT"))
same = ns["build"].__globals__["same"]                                                       # 実験87 の「同じものか」（名前の一部も同じとみなす）
sys.stdout.reconfigure(encoding="utf-8"); orig = ns["question"]
df = lambda w: max(1, sum(1 for nodes in T if any(same(w, x) for n in nodes for x in n["words"])))
rare = lambda w: math.log(len(T) / df(w))


def make(mode):
    def question(q):
        _, r = orig(q); stop = {"何", "誰", "どこ", "名前", "どんな", "最後"}; b = torch.zeros(DIM)
        for bb in q["bunsetsu"]:
            for w in bb["words"]:
                if not w[2].startswith(CONTENT) or w[1] in stop or w[0] in stop:
                    continue
                v = w[2].startswith("動詞") or (w[2].startswith("名詞") and any(x[1] == "為る" for x in bb["words"]))
                rr = rare(w[1])
                wt = {"A": 2.0 if v else 1.0, "B": (1 + rr) if v else 1.0, "C": 1 + rr, "D": (1 + 2 * rr) if v else (1 + rr)}[mode]
                b += wt * own([w[1]], None)
        return b, r
    return question


res = dict(exp="102")
for mode in "ABCD":
    ns["question"] = make(mode); R1 = run_all(Q1, A1); R2 = run_all(Q2, A2)
    res[mode] = dict(q20=sum(r["上から_ok"] for r in R1), q10=sum(r["上から_ok"] for r in R2), wrong=[r["q"] for r in R1 + R2 if not r["上から_ok"]])
    print(mode, res[mode]["q20"], res[mode]["q10"], res[mode]["wrong"], flush=True)
with open("results/exp102.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
