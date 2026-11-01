"""実験97: 重みをバッファ（質問）の側で変える（実験96 の続き、持ち主「うん、そしたらそれをお願い」2026-11-02）

実験96: 木の側に固定の重み（述語・主語を重く）を付けると悪くなった（27 → 18〜25）。手がかりの名詞が薄まるため。
ここでは、木はそのまま（重みなし）にして、バッファ（質問の手がかりの光）の中で、手がかりの語の重みを質問の種類で変える。
  動詞を重く: 質問の述語（見つける・渡す・歓迎する…）の名札を 2 倍。
  名詞を重く: 質問の名詞（太郎・亀・玉手箱…）の名札を 2 倍。
  質問の種類で切り替え: のは の質問・「誰」の質問（行った人を探す）は 述語 を 2 倍、ほか（何を・どこへ・どんな…）は 名詞 を 2 倍。
比べ: 重みなし（実験89 の「上から」）。浦島太郎の質問 20 + 使っていない 10。
結果: results/exp97.txt。実行: python experiments/exp97_buffer_weights.py
"""
import json, os, sys

import torch

spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
DIM, own, run_all, Q1, A1, Q2, A2, WH, CONTENT, RI, VOC = ns["DIM"], ns["own"], ns["run_all"], ns["Q1"], ns["A1"], ns["Q2"], ns["A2"], ns["WH"], ns["CONTENT"], ns["RI"], ns["VOC"]
sys.stdout.reconfigure(encoding="utf-8")
orig = ns["question"]


def make_question(mode):
    def question(q):
        txt = q["sentence"]; _, r = orig(q)
        stop = {"何", "誰", "どこ", "名前", "どんな", "最後"}
        cleft_or_who = ("のは" in txt) or ("誰" in txt)
        b = torch.zeros(DIM)
        for bb in q["bunsetsu"]:
            for w in bb["words"]:
                if not w[2].startswith(CONTENT) or w[1] in stop or w[0] in stop:
                    continue
                is_verb = w[2].startswith("動詞") or (w[2].startswith("名詞") and any(x[1] == "為る" for x in bb["words"]))   # 「到着する」のような 名詞 + する も述語
                wt = 1.0
                if mode == "動詞を重く" and is_verb:
                    wt = 2.0
                if mode == "名詞を重く" and not is_verb:
                    wt = 2.0
                if mode == "質問の種類で切り替え":
                    wt = 2.0 if (is_verb if cleft_or_who else not is_verb) else 1.0
                b += wt * own([w[1]], None)
        return b, r
    return question


res = dict(exp="97")
for mode in ("重みなし", "動詞を重く", "名詞を重く", "質問の種類で切り替え"):
    ns["question"] = orig if mode == "重みなし" else make_question(mode)
    R1 = run_all(Q1, A1); R2 = run_all(Q2, A2)
    res[mode] = dict(q20=sum(r["上から_ok"] for r in R1), q10=sum(r["上から_ok"] for r in R2), wrong=[r["q"] for r in R1 + R2 if not r["上から_ok"]])
    print(mode, "20 問", res[mode]["q20"], "未使用 10 問", res[mode]["q10"], "外れ", res[mode]["wrong"], flush=True)
with open("results/exp97.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
