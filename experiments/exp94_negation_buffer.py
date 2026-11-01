"""実験94: 否定をバッファに入れる（持ち主の 0〜3 のうち「2. 構文による属性の変化」）— はい・いいえで答える質問で、否定の向きを光で持てるか

着想: 2026-11-02 持ち主「構文による属性の変化（接頭語による not、and Not ぽい語板の学習など）→ バッファに反映」「否定は 1 − p にする、差分で引く、の考えでいい」。
      実験93 で、否定の「ない」は前後の形から板として分かれた。ここでは、否定を「向き」の光（+1 / −1）として節グレインとバッファに持たせる。
作り方（実験89 の「上から」の答え探しに足す）:
  節の向き: 述語の文節に、否定の助動詞（ない・ず・ぬ・ん）が付くたびに向きを反転（−1 を掛ける）。「無い」（形容詞）は「有る」+ 反転。
      「〜てはならない・〜てはいけない」（禁止）は、許しの向き −1、「〜てもよい・〜ていい」（許し）は +1。
  質問の向き: 質問の述語に同じ決まりを当てる（「ありましたか」= +1、「なかったですか」= −1）。
  答え: 質問の手がかりの光で文（幹）を選び、その文の述語の節のうち、手がかりの光といちばん重なる節を選ぶ。節の向き × 質問の向き = +1 なら「はい」、−1 なら「いいえ」
      （日本語の「〜なかったですか」への答え方: 質問の言い方どおりなら「はい」）。
  比べ: 向きを持たない（見つかれば いつも「はい」）。
限界: 否定の語・禁止と許しの言い回しの決まりは手で書いた小さな表（実験93 の板分けで「ない」を見つけ、向きの反転は決まり）。
質問: 浦島太郎のはい・いいえの質問 12（否定の文に当たるもの、否定の言い方の質問を含む）。
結果: results/exp94.txt、docs/negation_buffer_94.md。実行: python experiments/exp94_negation_buffer.py
"""
import json, os, sys

import torch

spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
T, S, cos, tag_of, own, DIM, VI = ns["T"], ns["S"], ns["cos"], ns["tag_of"], ns["own"], ns["DIM"], ns["VI"]
sys.stdout.reconfigure(encoding="utf-8")
Q = json.load(open("data/urashima_yesno_parse.json", encoding="utf-8")); A = [l.rstrip("\n").split("\t")[1] for l in open("data/urashima_yesno.tsv", encoding="utf-8")]
NEG_AUX = {"ない", "ず", "ぬ", "ん"}


def polarity(words):
    """文節の語（[表層, 正規形, 品詞]）から向き。否定の助動詞・無い で反転、禁止・許しの言い回し。返り値: (向き, 主な述語の語)。"""
    norms = [w[1] for w in words]; text = "".join(w[0] for w in words); pol = 1; pred = None
    for w in words:
        if w[2].startswith("助動詞") and w[1] in NEG_AUX:
            pol = -pol
        if w[1] == "無い" and w[2].startswith("形容詞"):
            pol = -pol; pred = pred or "有る"
        if pred is None and w[2].startswith(("動詞-一般", "動詞-非自立", "形容詞")) and w[1] not in ("為る", "居る", "成る", "おる"):
            pred = w[1]
    if pred is None and "居る" in norms:
        pred = "居る"
    if ("ては" in text or "では" in text) and ("ならない" in text or "いけない" in text or ("成る" in norms and pol == -1)):
        pol = -1                                                                             # 禁止: 〜してはならない → 許しの向き −1
    if ("ても" in text or "でも" in text) and ("良い" in norms or "いい" in text or "よい" in text):
        pol = 1                                                                              # 許し: 〜してもよい → +1
    return pol, pred


PRED_TAG = lambda p: tag_of(p) if p else torch.zeros(DIM)
rows = []
for q, gold in zip(Q, A):
    B = q["bunsetsu"]; root = next(b for b in B if b["head"] == -1)
    qpol, qpred = polarity(root["words"])
    if any(w[1] == "言う" for w in root["words"]):                                           # 「〜と言いましたか」: 言った中身（引用の節）の向きを見る
        inner = [b for b in B if b["head"] == root["id"] and b["text"].rstrip("、").endswith("と")]
        if inner:
            qpol, qpred = polarity(inner[0]["words"])
    stop = {"何", "誰", "どこ", "名前", "どんな", "最後"}
    b = torch.zeros(DIM)
    for bb in B:
        ws = [w[1] for w in bb["words"] if w[2].startswith(ns["CONTENT"]) and w[1] not in stop]
        ws = ["有る" if w == "無い" else w for w in ws]
        b += own(ws, None)
    si = max(range(len(T)), key=lambda i: cos(next(m for m in T[i] if m["head"] == -1)["G"], b))
    best = None
    for n in T[si]:
        words = S[si]["bunsetsu"][n["id"]]["words"]
        pol, pred = polarity(words)
        if pred is None:
            continue
        sc = cos(n["G"], b) + 1.0 * float(PRED_TAG(pred)[: len(VI)] @ PRED_TAG(qpred)[: len(VI)] > 0)   # 述語が同じ（名札が重なる）なら強く
        if best is None or sc > best[0]:
            best = (sc, n, pol, pred)
    _, n, pol, pred = best
    ans = "はい" if pol * qpol > 0 else "いいえ"
    rows.append(dict(q=q["sentence"], gold=gold, sent=si + 1, node=n["text"], node_pol=pol, q_pol=qpol, ans=ans, ok=ans == gold, base_ok=gold == "はい"))
res = dict(exp="94", correct=sum(r["ok"] for r in rows), base_correct=sum(r["base_ok"] for r in rows), n=len(rows), rows=rows)
with open("results/exp94.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
L = ["# 否定をバッファに入れる（実験94、浦島太郎のはい・いいえの質問）", "", "作り方: experiments/exp94_negation_buffer.py（実験89 の上からの答え探し + 否定の向き）。",
     f"正解: **{res['correct']} / {res['n']}**（向きを持たない = いつも「はい」なら {res['base_correct']} / {res['n']}）。", "",
     "| 質問 | 正解 | 文 | 見つけた述語の節 | 節の向き | 質問の向き | 答え | |", "|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['q']} | {r['gold']} | {r['sent']} | {r['node']} | {'+' if r['node_pol'] > 0 else '−'} | {'+' if r['q_pol'] > 0 else '−'} | {r['ans']} | {'○' if r['ok'] else '×'} |")
open("docs/negation_buffer_94.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
