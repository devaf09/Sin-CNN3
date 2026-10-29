"""実験72: 過飽和の判定と、合成の語板の候補（持ち主の案 2026-10-29）

持ち主: 「〜の〜 のように属性の相手が多すぎる場合は、前か後ろの語をくっつけて新しい語板にする。それでも多ければさらにつなげる。
         属性層が過飽和になったら、なるべく新しい語板が属性を持っていかない方向の語とつなげる。過飽和には式が要る（最も強い点が標準偏差の表を超えたら、など）」。
材料: 観点を増やして数え直した材料をまとめたもの（非公開フォルダの data/views/merged2.pkl。会話 × 100、R15・R18 × 0.1）。
      観点 9 = どんな・を・が・で・に・の前（A の [語]）・の後（[語] の B）・と・は（形容詞）。
過飽和の判定（観点ごと、語板が見る形 = 観点ごとの上位 47 値 + その他）:
    p = その観点の値の割合（47 値 + その他）。いちばん強い点 p1 が、ほかの点の平均 + k × 標準偏差を超えなければ「突出がない」、
    または「その他」の割合が 0.5 を超えれば、過飽和。k = 2・3・4 で数を比べる。
合成の語板の候補:
  (a) 過飽和の案（持ち主）: 「の前」「の後」のどちらかが過飽和の名詞について、前の相手（X の [語]）と後ろの相手（[語] の Y）それぞれ上位 3 つを候補にする。
      どちらの向きにつなげるか（新しい語板が持っていく属性が少ない向き）は、候補を 1 語として数え直したあとで決める。
  (b) 相互情報量の案: 「A の B」の組のうち、たまたまより多く一緒に出る（相互情報量が大きい）もの。数（重み付き）が 200 以上。上位 3000 組。
出力: results/exp72.txt（数）、非公開フォルダの data/views/compound_candidates.json（候補の組）、docs/saturation_72.md（一覧の例）
実行: python experiments/exp72_saturation.py
"""
import collections, json, math, os, pickle, sys, time

sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time()
D = pickle.load(open("D:/AI_git/Sin-CNN3_private/data/views/merged2.pkl", "rb"))
TR = D["train"]; VIEWS = ["どんな", "を", "が", "で", "に", "の前", "の後", "と", "は"]
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
allv = {v: collections.Counter() for v in VIEWS}
for n, vs in TR.items():
    for v, c in vs.items():
        if v in allv:
            allv[v].update(c)
TOP = {v: set(x for x, _ in allv[v].most_common(47)) for v in VIEWS}                      # 語板が別々に覚える値（観点ごとの上位 47）
print("読んだ", len(TR), "名詞", f"{time.time() - t0:.0f} 秒", flush=True)


def slots(c, v, own=True):
    """観点 v の数 c を、47 個の値 + その他 の 48 個の割合にする。
    own = True: その語自身の多い順 47 個（語板が 1 語で持てる量。「〜の〜」「〜と〜」は相手が語ごとにばらばらなので、全体の上位 47 では測れない）。
    own = False: 全体の上位 47 個（実験58〜61 の語板の形）。最初はこちらで測り、「の前・の後・と」はほぼ全部の語が「その他」過多になって、判定にならなかった。"""
    tot = sum(c.values())
    if tot <= 0:
        return None
    keys = [x for x, _ in collections.Counter(c).most_common(47)] if own else sorted(TOP[v])
    vals = [c.get(x, 0) / tot for x in keys] + [0.0] * (47 - len(keys)); other = 1 - sum(vals)
    return vals + [other]


def saturated(p, k):
    p1 = max(p[:-1]); rest = sorted(p[:-1], reverse=True)[1:]
    m = sum(rest) / len(rest); sd = math.sqrt(sum((x - m) ** 2 for x in rest) / len(rest))
    return (p1 <= m + k * sd) or (p[-1] > 0.5)


NOUNS = [n for n in sorted(D["noun_total"], key=D["noun_total"].get, reverse=True) if n not in SKIP][:50000]
res = dict(exp="72", nouns=len(NOUNS)); sat = {}
for k in (2, 3, 4):
    cnt = collections.Counter(); ex = collections.defaultdict(list)
    for n in NOUNS:
        for v in VIEWS:
            c = TR[n].get(v)
            if not c or sum(c.values()) < 50:                                                 # 数の少ない観点は判定しない
                continue
            p = slots(c, v)
            if p and saturated(p, k):
                cnt[v] += 1
                if len(ex[v]) < 15:
                    ex[v].append(n)
                if k == 3:
                    sat.setdefault(n, []).append(v)
    res[f"k={k}"] = dict(cnt); res[f"k={k} 例"] = {v: e for v, e in ex.items()}
    print("k", k, dict(cnt), flush=True)
# (a) 過飽和の案: 「の前」「の後」が過飽和（k = 3）の名詞のうち、数の多い 1000 語の、前の相手・後ろの相手の上位 3
import re
okw = lambda w: bool(re.fullmatch(r"[ぁ-んァ-ヶー一-龠々]+", w))                              # 記号・英数字だけの語は除く
cand_a = []
satn = [n for n in NOUNS if n in sat and ("の前" in sat[n] or "の後" in sat[n]) and okw(n)][:1000]
for n in satn:
    if True:
        for x, _ in collections.Counter({k: m for k, m in TR[n].get("の前", {}).items() if okw(k)}).most_common(3):
            cand_a.append((x, n))                                                           # X の [語]（前につなげる）
        for y, _ in collections.Counter({k: m for k, m in TR[n].get("の後", {}).items() if okw(k)}).most_common(3):
            cand_a.append((n, y))                                                           # [語] の Y（後ろにつなげる）
cand_a = sorted(set(cand_a))
# (b) 相互情報量の案
front = collections.Counter(); back = collections.Counter(); pair = collections.Counter()
for n, vs in TR.items():
    for y, m in vs.get("の後", {}).items():
        pair[(n, y)] += m; front[n] += m; back[y] += m
Z = sum(pair.values())
pmi = {ab: math.log(m * Z / (front[ab[0]] * back[ab[1]])) for ab, m in pair.items() if m >= 200 and ab[0] not in SKIP and ab[1] not in SKIP and okw(ab[0]) and okw(ab[1])}
cand_b = [ab for ab, _ in sorted(pmi.items(), key=lambda t: -t[1] * math.log(pair[t[0]]))[:3000]]
res["候補 (a) 過飽和"] = len(cand_a); res["候補 (b) 相互情報量"] = len(cand_b); res["両方に入る"] = len(set(cand_a) & set(cand_b))
json.dump(dict(a=cand_a, b=cand_b), open("D:/AI_git/Sin-CNN3_private/data/views/compound_candidates.json", "w", encoding="utf-8"), ensure_ascii=False)
lines = ["# 過飽和の判定と合成の語板の候補（実験72）", "", "観点ごとに、その語の多い順 47 個の値 + その他 にして、いちばん強い点がほかの点の平均 + k × 標準偏差を超えない、または「その他」（48 番目以降）が半分を超えると過飽和。",
         "（材料: 観点を増やして数え直したもの、会話 × 100・R15・R18 × 0.1。名詞 5 万語、数の少ない観点（50 未満）は判定しない）", "",
         "| k | " + " | ".join(VIEWS) + " |", "|---|" + "---|" * len(VIEWS)]
for k in (2, 3, 4):
    lines.append(f"| {k} | " + " | ".join(str(res[f'k={k}'].get(v, 0)) for v in VIEWS) + " |")
lines += ["", "## 過飽和の例（k = 3、観点ごとに 15 語）", ""] + [f"- **{v}**: {'・'.join(e)}" for v, e in res["k=3 例"].items()]
lines += ["", f"## 合成の語板の候補", "", f"- (a) 過飽和の案（「の前」「の後」が過飽和の名詞のうち数の多い 1000 語の、前後の相手の上位 3）: {len(cand_a)} 組。例: " + "・".join(f"{a}の{b}" for a, b in cand_a[:30]),
          f"- (b) 相互情報量の案（たまたまより多く一緒に出る組、上位 3000）: {len(cand_b)} 組。例: " + "・".join(f"{a}の{b}" for a, b in cand_b[:30]),
          f"- 両方に入る組: {res['両方に入る']}"]
open("docs/saturation_72.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
res["sec"] = time.time() - t0
with open("results/exp72.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print("\n".join(lines))
