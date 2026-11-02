"""実験115: 返事を選ぶ（テストボット02 の土台、会話の物差し）。持ち主「なるほど、ではそれで！」2026-11-02

物差し: 本物の会話を途中で止め、正しい次の発言を 10 個の候補（正解 1 + ほかの会話からの発言 9）から選べるか（1 位の正答・MRR。でたらめ 10%）。
  人の採点も代役の AI も要らない、毎回同じに測れる会話の物差し。
選び方（ニューラルなし、これまでの部品で）:
  A 語の重なり: 直前の発言と候補で、同じ中身の語がいくつあるか（話に出た語のキャッシュ、実験81 の考え）。
  B 次の発言の共鳴: 教えの会話で「ある発言に x があると、次の発言に y が出やすい」を数えた PMI（実験77 の「次の文」の観点を数え上げで）。
     直前の発言の語 × 候補の語 の共鳴の平均。
  C A + B。
  D C + 文脈を 3 発言に広げる（古い発言ほど 0.5 倍）。
  E D + 口調の合い: 直前の相手の発言が丁寧（です・ます）なら丁寧な候補、くだけていればくだけた候補を少し上げる（話者関係の最初の手がかり）。
材料: Sin-CNN3_private/data/text/conv_all.jsonl の RealPersonaChat（雑談、CC BY-SA 4.0）と JMultiWOZ（案内の対話、CC BY-SA 4.0）。会話ごとに教え 9 : 試し 1。
  表示: RealPersonaChat（nu-dialogue、CC BY-SA 4.0）、JMultiWOZ（nu-dialogue、CC BY-SA 4.0）。
結果: results/exp115.txt、docs/testbot02.md、図 docs/figs/response_selection_115.svg。実行: python experiments/exp115_response_selection.py
"""
import json, math, random, re, sys, zlib
from collections import Counter, defaultdict

from sudachipy import Dictionary

sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"
tok = Dictionary(dict="core").tokenizer()
CONTENT = ("名詞", "動詞", "形容詞", "形状詞", "副詞")
STOP = {"為る", "居る", "成る", "有る", "言う", "思う", "事", "物", "の", "其れ", "此れ", "私", "貴方", "方", "そう", "然う", "ちょっと", "一寸", "良い", "とても", "本当", "凄い"}


def analyze(text):
    ms = tok.tokenize(text)
    words = {m.normalized_form() for m in ms if m.part_of_speech()[0] in CONTENT and m.normalized_form() not in STOP}
    polite = any(m.normalized_form() in ("です", "ます", "御座る") for m in ms)
    return words, polite


convs = {"train": [], "test": []}
for l in open(f"{PRIV}/data/text/conv_all.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if d["id"].startswith("aozora"):
        continue
    turns = [t.strip() for t in d["text"].split("\n") if t.strip()]
    if len(turns) < 4:
        continue
    convs["test" if zlib.crc32(d["id"].encode()) % 10 == 0 else "train"].append((d["id"].split(":")[0], turns))
print("会話 教え", len(convs["train"]), "試し", len(convs["test"]), flush=True)
# ---- 教え: 次の発言の共鳴を数える ----
cw = Counter(); cnext = Counter(); cpair = Counter(); npairs = 0
for src, turns in convs["train"]:
    A = [analyze(t)[0] for t in turns]
    for i in range(len(A) - 1):
        npairs += 1
        for x in A[i]:
            cw[x] += 1
        for y in A[i + 1]:
            cnext[y] += 1
        for x in A[i]:
            for y in A[i + 1]:
                cpair[(x, y)] += 1
print("発言の組", npairs, "語の組", len(cpair), flush=True)


def pmi(x, y):
    c = cpair.get((x, y), 0)
    if c < 2:
        return 0.0
    return max(0.0, math.log(c * npairs / (cw[x] * cnext[y])))


# ---- 試し: 10 択 ----
rng = random.Random(0)
pool = [(src, t) for src, turns in convs["test"] for t in turns]
cases = []
for ci, (src, turns) in enumerate(convs["test"]):
    for i in range(3, len(turns), 4):                                                          # 会話ごとに数か所で止める
        negs = []
        while len(negs) < 9:
            s2, t2 = rng.choice(pool)
            if t2 != turns[i] and t2 not in turns:
                negs.append(t2)
        same = [t for j, t in enumerate(turns) if abs(j - i) > 1 and t != turns[i]]          # むずかしい版: 同じ会話のほかの発言（話題は同じ、流れだけが違う）
        cases.append((src, turns[max(0, i - 3):i], turns[i], negs, rng.sample(same, min(9, len(same)))))
rng.shuffle(cases); cases = cases[:3000]                                                       # 雑談と案内の対話が両方入るように混ぜる
print("試しの問", len(cases), flush=True)
cache = {}
an = lambda t: cache.setdefault(t, analyze(t))


def score(ctx, cand, how):
    cw_, cp = an(cand)
    sc = 0.0
    hist = list(reversed(ctx)) if how in ("D", "E") else [ctx[-1]]
    for k, u in enumerate(hist):
        uw, up = an(u); wk = 0.5 ** k
        if how in ("A", "C", "D", "E"):
            sc += wk * len(uw & cw_) / math.sqrt(len(cw_) + 1)                                # 同じ語（キャッシュ）
        if how in ("B", "C", "D", "E") and uw and cw_:
            sc += wk * sum(pmi(x, y) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_))   # 次の発言の共鳴
    if how == "E":
        sc += 0.3 if an(ctx[-1])[1] == cp else 0.0                                              # 口調の合い（直前の相手の発言と）
    return sc


res = dict(exp="115", n_cases=len(cases), runs=[])
for hard in (False, True):
  for how, name in (("R", "でたらめ"), ("A", "語の重なり"), ("B", "次の発言の共鳴"), ("C", "重なり + 共鳴"), ("D", "+ 文脈 3 発言"), ("E", "+ 口調の合い")):
      top1 = rr = 0.0; by = defaultdict(lambda: [0, 0])
      for src, ctx, gold, negs, same in cases:
          cands = [gold] + (same if hard else negs)
          if how == "R":
              scs = [rng.random() for _ in cands]
          else:
              scs = [score(ctx, c, how) + 1e-6 * rng.random() for c in cands]
          rank = 1 + sum(s > scs[0] for s in scs[1:])
          top1 += rank == 1; rr += 1 / rank; by[src][0] += rank == 1; by[src][1] += 1
      r = dict(hard=hard, how=how, name=name, n_cands=10, top1=top1 / len(cases), mrr=rr / len(cases), by_source={k: round(v[0] / v[1], 3) for k, v in by.items()})
      res["runs"].append(r); print("同じ会話から" if hard else "ほかの会話から", name, "1 位", round(r["top1"], 3), "MRR", round(r["mrr"], 3), r["by_source"], flush=True)
with open("results/exp115.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
