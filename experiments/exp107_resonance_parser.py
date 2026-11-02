"""実験107: 共鳴パーシング（別の AI の助言の提案 1、GiNZA の係り先を使わずに、自分で係り受けの矢印を張る）

考え: 文節 i の「尾」（最後の機能語: が・を・の・て…）は「右のどんな文節に係りたいか」という要求の波を出す。
      文節 j の「型」（中身の品詞 + j の尾）は受け入れの波を出す。要求と受け入れの合い具合（共鳴）= PMI(尾_i, 型_j)。
      日本語は係り先が右にあり、矢印は交わらない。i を右から順に、i+1 から辿れる係り先の鎖の上で、共鳴 − α log(鎖の何番目か) が最大の所へ係る。
共鳴の覚え方（自己構造化）:
  0 回目: 答えなしで、同じ文の中の「右にある全部の文節」との組を 1/距離 の重みで数える（ただ一緒に出るだけ）。
  1〜5 回目: 自分の今の共鳴で教えの文を解析し、自分が張った矢印だけで数え直す（自分の木から自分を学び直す = 答えなしの繰り返し）。
比べ: すぐ右に係る / 最後の文節に係る / 手の決まり（の・連体の語 → 鎖の上の一番近い名詞、ほか → 一番近い述語）/
      答えあり（GiNZA の係り先で数えた共鳴、上限の目安）。
文節の切り方は GiNZA のものを使う（係り先は使わない）。文節の切り方も自分でやるのは次の段。
評価: 試しの文で、最後の文節以外の文節の係り先が GiNZA と一致する割合（係り先の正答）。
材料: Sin-CNN3_private/data/views/sentence_trees.json（会話・青空文庫 1.5 万文、Web 小説は使わない）。
結果: results/exp107.txt、docs/resonance_parser_107.md、図 docs/figs/resonance_parser_107.svg。実行: python experiments/exp107_resonance_parser.py
"""
import json, math, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"
FUNC = ("助詞", "助動詞", "補助記号")
S = json.load(open(f"{PRIV}/data/views/sentence_trees.json", encoding="utf-8"))
tr = [s for s in S if s["split"] == "train"]; te = [s for s in S if s["split"] == "test"]


def feats(b):
    """文節の 尾（要求の側）と 型（受け入れの側）。"""
    core = next((w for w in b["w"] if not w[2].startswith(FUNC)), b["w"][0])
    cpos = core[2].split("-")[0]
    funcs = [w[1] for w in b["w"] if w[2].startswith(("助詞", "助動詞"))]
    comma = "、" if b["w"][-1][0] in ("、", "，") else ""
    tail = (funcs[-1] if funcs else "∅" + cpos) + comma
    end = funcs[-1] if funcs else "∅"
    return tail, cpos + "|" + end, cpos


for s in S:
    for b in s["b"]:
        b["tail"], b["type"], b["cpos"] = feats(b)
    n = len(s["b"])
    for i, b in enumerate(s["b"]):
        if b["head"] <= i and i < n - 1:                                                       # GiNZA が左や自分に係らせた所（まれ）は、評価から外す印
            b["skip"] = True


def chain(heads, i, n):
    """i が係れる先: i+1、i+1 の係り先、その係り先 … （矢印が交わらない所だけ）。"""
    out, j = [], i + 1
    while j < n:
        out.append(j)
        if j == n - 1 or heads[j] is None:
            break
        j = heads[j]
    return out


def parse(s, score, alpha):
    n = len(s["b"]); heads = [None] * n
    for i in range(n - 2, -1, -1):
        cand = chain(heads, i, n)
        heads[i] = max(((score(s["b"][i], s["b"][j]) - alpha * math.log(1 + r), j) for r, j in enumerate(cand)))[1]
    heads[n - 1] = n - 1
    return heads


def uas(data, score, alpha):
    ok = tot = 0
    for s in data:
        h = parse(s, score, alpha)
        for i, b in enumerate(s["b"][:-1]):
            if b.get("skip"):
                continue
            ok += h[i] == b["head"]; tot += 1
    return ok / tot


def pmi_from(pairs):
    """数えた組 (尾, 型, 重み) から 共鳴 = PMI（足し 0.5 でならす）。"""
    c = Counter(); ct = Counter(); cy = Counter(); tot = 0.0
    for t, y, w in pairs:
        c[(t, y)] += w; ct[t] += w; cy[y] += w; tot += w
    ny = len(cy)

    def score(bi, bj):
        t, y = bi["tail"], bj["type"]
        return math.log((c[(t, y)] + 0.5) * tot / ((ct[t] + 0.5 * ny) * (cy[y] + 0.5)))
    return score


def pairs_cooc(data):
    for s in data:
        B = s["b"]
        for i in range(len(B) - 1):
            for j in range(i + 1, len(B)):
                yield B[i]["tail"], B[j]["type"], 1.0 / (j - i)


def pairs_parsed(data, score, alpha):
    for s in data:
        h = parse(s, score, alpha)
        for i in range(len(s["b"]) - 1):
            yield s["b"][i]["tail"], s["b"][h[i]]["type"], 1.0


def pairs_gold(data):
    for s in data:
        for i, b in enumerate(s["b"][:-1]):
            if not b.get("skip"):
                yield b["tail"], s["b"][b["head"]]["type"], 1.0


NOUNISH = ("名詞", "代名詞")
PRED = ("動詞", "形容詞", "形状詞")


def rule(bi, bj):
    """手の決まり: の・連体詞・連体の語 → 名詞に、ほか → 述語に。"""
    want_noun = bi["tail"].rstrip("、") in ("の",) or bi["cpos"] in ("連体詞",) or (bi["tail"].startswith("∅形容詞") and not bi["tail"].endswith("、"))
    if want_noun:
        return 1.0 if bj["cpos"] in NOUNISH else 0.0
    return 1.0 if bj["cpos"] in PRED or bj["type"].split("|")[1] in ("だ", "です") else 0.0


res = dict(exp="107", n_train=len(tr), n_test=len(te), runs=[])
log = lambda name, v, **kw: (res["runs"].append(dict(name=name, uas=v, **kw)), print(name, round(v, 4), kw, flush=True))
log("すぐ右", uas(te, lambda a, b: 0.0, 1.0))
log("最後の文節", uas(te, lambda a, b: 0.0, -1.0))
log("手の決まり", uas(te, rule, 0.1))
gold = pmi_from(pairs_gold(tr))
for al in (0.5, 1.0, 2.0):
    log("答えあり（GiNZA で数えた共鳴）", uas(te, gold, al), alpha=al)
ALPHA = 1.0
sc = pmi_from(pairs_cooc(tr))
log("答えなし 0 回目（一緒に出るだけ）", uas(te, sc, ALPHA), round=0, alpha=ALPHA)
for it in range(1, 6):
    sc = pmi_from(pairs_parsed(tr, sc, ALPHA))                                                  # 自分の木から自分を学び直す
    log(f"答えなし {it} 回目（自分の木で数え直し）", uas(te, sc, ALPHA), round=it, alpha=ALPHA)
# 例: 学んだ共鳴の強い組（尾 → 型）
c = Counter(); ct = Counter()
for t, y, w in pairs_parsed(tr, sc, ALPHA):
    c[(t, y)] += w; ct[t] += w
res["top_pairs"] = {t: [(y, round(c[(t, y)] / ct[t], 2)) for (tt, y), _ in sorted(((k, v) for k, v in c.items() if k[0] == t), key=lambda kv: -kv[1])[:3]]
                    for t in ["が", "を", "に", "で", "は", "の", "て", "と", "∅形容詞", "∅副詞", "∅連体詞", "た"] if ct[t]}
print(json.dumps(res["top_pairs"], ensure_ascii=False))

# ---- 追加: 型を「中身の品詞だけ」に粗くし、点を P(型 | 尾)（割合）にした版。答えあり・答えなし（すぐ右から始めて自分の木で数え直す）・満ち（同じ格を 2 つ受けない）----
CASE = {"が", "を", "に", "で", "は", "と", "へ", "から"}


def prob_from(pairs):
    c = Counter(); ct = Counter(); cy = set()
    for t, y, w in pairs:
        c[(t, y)] += w; ct[t] += w; cy.add(y)
    ny = len(cy)
    return lambda bi, bj: math.log((c[(bi["tail"], bj["cpos"])] + 0.1) / (ct[bi["tail"]] + 0.1 * ny))


def parse_sat(s, score, alpha, sat):
    n = len(s["b"]); heads = [None] * n; got = [set() for _ in range(n)]
    for i in range(n - 2, -1, -1):
        t = s["b"][i]["tail"].rstrip("、")
        heads[i] = max(((score(s["b"][i], s["b"][j]) - alpha * math.log(1 + r) - (sat if t in CASE and t in got[j] else 0), j) for r, j in enumerate(chain(heads, i, n))))[1]
        got[heads[i]].add(t)
    heads[n - 1] = n - 1
    return heads


def uas_sat(data, score, alpha, sat):
    ok = tot = 0
    for s in data:
        h = parse_sat(s, score, alpha, sat)
        for i, b in enumerate(s["b"][:-1]):
            if not b.get("skip"):
                ok += h[i] == b["head"]; tot += 1
    return ok / tot


gold2 = prob_from((b["tail"], s["b"][b["head"]]["cpos"], 1.0) for s in tr for b in s["b"][:-1] if not b.get("skip"))
for sat in (0, 3):
    log("答えあり（品詞だけ・割合）", uas_sat(te, gold2, 0.3, sat), sat=sat)
for sat in (0, 3):
    sc2 = prob_from((s["b"][i]["tail"], s["b"][i + 1]["cpos"], 1.0) for s in tr for i in range(len(s["b"]) - 1))   # すぐ右から始める
    curve = [uas_sat(te, sc2, 1.0, sat)]
    for it in range(5):
        cur = sc2
        sc2 = prob_from((s["b"][i]["tail"], s["b"][h[i]]["cpos"], 1.0) for s in tr for h in [parse_sat(s, cur, 1.0, sat)] for i in range(len(s["b"]) - 1))
        curve.append(uas_sat(te, sc2, 1.0, sat))
    log("答えなし（品詞だけ・割合、すぐ右から自分の木で数え直し）", curve[-1], sat=sat, curve=[round(x, 4) for x in curve])
log("手の決まり + 満ち", uas_sat(te, rule, 0.1, 3), sat=3)
with open("results/exp107.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
