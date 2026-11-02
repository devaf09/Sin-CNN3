"""実験119: 格逆算グレインの最初の形。会話の中で、言われている格をわざと消し、前の発言の流れと述語から当てるように学ぶ（持ち主の案 2026-11-02）

持ち主: 「格が抜いても話が維持できる会話から格を抜いて学習して、逆回しで想定することってできないかな？」
作り方（答えは消す前の元の発言そのもの、人の手も他の AI も使わない）:
  1. 会話の発言の中から「名詞 + が・を・に → その後の最初の動詞」の組を探す（例: ラーメン を 食べる）。
  2. 名詞を消し、残りから当てる。候補 = 前の 3 発言とその発言の中の名詞（会話に出ている語）。
  3. 本当の省略に近い場合（消した名詞が前の 3 発言にも出ている）を中心に測る。人は「もう話に出ていて分かる」時に格を省くため。
当て方:
  A 近さ: いちばん最近に出た名詞。
  B 述語の格の好み: 教えの会話で数えた P(名詞 | 動詞, 格)（食べる の を には 食べ物が来やすい）。
  C A + B。
  D C + 発言どうしの共鳴（実験115 の次の発言の共鳴、前の発言の語との PMI）。
材料: RealPersonaChat・JMultiWOZ（nu-dialogue、CC BY-SA 4.0）。会話ごとに教え 9 : 試し 1（実験115 と同じ分け方）。
結果: results/exp119.txt、docs/testbot02.md に追記。実行: python experiments/exp119_case_recovery.py
"""
import json, math, random, sys
from collections import Counter, defaultdict

src = open("experiments/exp115_response_selection.py", encoding="utf-8").read().split("# ---- 試し: 10 択 ----")[0]
exec(src)                                                                                      # 材料・解析・次の発言の共鳴（実験115）
CASE = ("が", "を", "に")


def nouns_and_frames(text):
    """発言の 名詞の並び と、（名詞, 格, 動詞）の組。"""
    ms = list(tok.tokenize(text)); nouns = []; frames = []
    for i, m in enumerate(ms):
        pos = m.part_of_speech()
        if pos[0] == "名詞" and pos[1] in ("普通名詞", "固有名詞") and len(m.surface()) >= 2:
            nouns.append(m.normalized_form())
            if i + 1 < len(ms) and ms[i + 1].surface() in CASE and ms[i + 1].part_of_speech()[1] == "格助詞":
                v = next((x for x in ms[i + 2:i + 6] if x.part_of_speech()[0] == "動詞" and x.part_of_speech()[1] != "非自立可能"), None)
                if v is not None and m.normalized_form() not in STOP:
                    frames.append((m.normalized_form(), ms[i + 1].surface(), v.normalized_form()))
    return nouns, frames


# ---- 教え: 述語の格の好み P(名詞 | 動詞, 格) ----
F = defaultdict(Counter); NF = Counter()
for s, turns in convs["train"]:
    for t in turns:
        ns, fr = nouns_and_frames(t)
        for n, c, v in fr:
            F[(v, c)][n] += 1
        for n in ns:
            NF[n] += 1
totN = sum(NF.values())
print("動詞と格の組", len(F), flush=True)


def frame_logp(n, v, c):
    cnt = F.get((v, c)); tot = sum(cnt.values()) if cnt else 0
    pn = (NF[n] + 1) / (totN + len(NF))
    return math.log(((cnt[n] if cnt else 0) + 5 * pn) / (tot + 5))                           # 少ない組は全体の割合でならす


# ---- 試し ----
cases = []
for s, turns in convs["test"]:
    seen = []                                                                                  # 前の発言の名詞（新しい順に後ろへ）
    for k, t in enumerate(turns):
        ns, fr = nouns_and_frames(t)
        prev = [n for u in turns[max(0, k - 3):k] for n in nouns_and_frames(u)[0]]
        for n, c, v in fr:
            cands = list(dict.fromkeys(reversed(prev + [x for x in ns if x != n])))          # 新しい順、重なりなし
            if n in prev and len(cands) >= 2:
                cases.append(dict(n=n, c=c, v=v, cands=cands, ctx=turns[max(0, k - 3):k]))
print("試しの問（消した名詞が前の 3 発言にも出ている）", len(cases), flush=True)
cache = {}
an = lambda t: cache.setdefault(t, analyze(t))


def pick(cs, how):
    best = None
    for r, x in enumerate(cs["cands"]):
        sc = 0.0
        if how in ("A", "C", "D"):
            sc += -0.5 * math.log(1 + r)                                                       # 近いほど
        if how in ("B", "C", "D"):
            sc += frame_logp(x, cs["v"], cs["c"])
        if how == "D":
            uw = set().union(*[an(u)[0] for u in cs["ctx"][-1:]]) if cs["ctx"] else set()
            sc += 0.3 * sum(pmi(y, x) for y in uw) / math.sqrt(len(uw) + 1)
        if best is None or sc > best[0]:
            best = (sc, x)
    return best[1]


res = dict(exp="119", n=len(cases), runs=[], chance=sum(1 / len(c["cands"]) for c in cases) / len(cases))
print("でたらめ", round(res["chance"], 3), flush=True)
for how, name in (("A", "近さ"), ("B", "述語の格の好み"), ("C", "近さ + 格の好み"), ("D", "+ 発言どうしの共鳴")):
    ok = Counter(); n = Counter()
    for cs in cases:
        g = pick(cs, how) == cs["n"]; ok[cs["c"]] += g; n[cs["c"]] += 1; ok["全体"] += g; n["全体"] += 1
    r = dict(how=how, name=name, acc={k: round(ok[k] / n[k], 3) for k in n}, n={k: n[k] for k in n})
    res["runs"].append(r); print(name, r["acc"], flush=True)
# 例（逆回し）: 格の好みの上位（述語と格 → 来やすい名詞）
res["frames"] = {f"{v} {c}": [x for x, _ in F[(v, c)].most_common(6)] for v, c in (("食べる", "を"), ("行く", "に"), ("見る", "を"), ("飼う", "を"), ("乗る", "に"), ("読む", "を")) if F.get((v, c))}
print(res["frames"])
with open("results/exp119.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
