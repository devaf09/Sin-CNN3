"""テストボット02: 返事を選ぶ会話ボット（実験115 の選び方 C: 語の重なり + 次の発言の共鳴）+ 結びは基本丁寧（実験117 の付け替え）
+ 感情 → 内容 → 質問 の型（バックトラッキング = 相手の述語を「〜んですね」で返す、持ち主が調べた会話継続のテクニック）。
持ち主との方針（2026-11-02）: いきなり文を一から作らず、会話の材料の中から返事を選ぶ。生成は少しずつ置き換える。
返事の候補: RealPersonaChat の雑談の発言（nu-dialogue、CC BY-SA 4.0）。案内の対話（JMultiWOZ）は雑談に合わないので候補から外す。
  手順: 1. 相手の発言の中身の語 → 2. 共鳴の強い「次に出やすい語」を引く（逆引きの索引）→ 3. その語を含む発言を候補にする
       → 4. 実験115 の点（語の重なり + 共鳴）で並べる → 5. 口調（相手が丁寧なら丁寧、くだけていればくだけた）が合うものを優先 → 上位から 1 つ。
  「自信」: 1 位の点が低いとき（共鳴の手がかりが無いとき）は「よく分からない」と正直に返す（実験62 の考えの最初の形）。
実行: python tools/testbot02.py "発言 1" "発言 2" …（引数が無ければ、決まった例で動かす）
"""
import json, math, random, sys, zlib
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.argv, ARGS = [sys.argv[0]], sys.argv[1:]
src = open("experiments/exp115_response_selection.py", encoding="utf-8").read().split("# ---- 試し: 10 択 ----")[0]
exec(src)                                                                                      # 材料・解析・共鳴の数え上げ（実験115）
# ---- 候補の発言と、逆引きの索引（語 → その語を含む発言）----
POOL = [t for s, turns in convs["train"] if s == "real-persona-chat" for t in turns if 4 <= len(t) <= 60]
POOL = list(dict.fromkeys(POOL))
INV = defaultdict(list)
for k, t in enumerate(POOL):
    for w in analyze(t)[0]:
        INV[w].append(k)
NEXT = defaultdict(list)                                                                       # 速くするため、先に語ごとの上位を作る
for (x, y), c in cpair.items():
    if c >= 3:
        NEXT[x].append((c, y))
for x in NEXT:
    NEXT[x] = [y for c, y in sorted(NEXT[x], reverse=True)[:60]]
print("候補の発言", len(POOL), flush=True)


def candidates(history):
    uw, up = analyze(history[-1])
    keys = set(uw) | {y for x in uw for y in NEXT.get(x, [])[:40]}
    cand = Counter()
    for y in keys:
        for k in INV.get(y, [])[:3000]:
            cand[k] += 1
    ids = [k for k, _ in cand.most_common(400)]
    if not ids:
        return []
    scored = []
    for k in ids:
        t = POOL[k]
        if t in history:
            continue
        cw_, cp = analyze(t)
        f_over = len(uw & cw_) / math.sqrt(len(cw_) + 1)
        f_res = sum(pmi(x, y) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_)) if uw and cw_ else 0.0
        f_tone = 1.0 if cp == up else 0.0
        scored.append((f_over + f_res + 0.3 * f_tone, t, (f_over, f_res, f_tone)))            # 点・発言・手がかり（重なり・共鳴・口調）
    scored.sort(key=lambda x: -x[0])
    return scored


def ensemble(scored, n=10, seed=0):
    """持ち主の案: 選ぶ重みを少しずつ揺らして 10 通り選び、票の多い返事を選ぶ。票の割合 = 自信。"""
    rng = random.Random(seed); votes = Counter()
    for _ in range(n):
        w = [math.exp(rng.gauss(0, 0.4)), math.exp(rng.gauss(0, 0.4)), 0.3 * math.exp(rng.gauss(0, 0.4))]
        best = max(scored, key=lambda x: w[0] * x[2][0] + w[1] * x[2][1] + w[2] * x[2][2])
        votes[best[1]] += 1
    t, v = votes.most_common(1)[0]
    return t, v / n


# ---- 結びの付け替え（実験117）と、バックトラッキング（持ち主が調べた会話継続のテクニック 2026-11-02）----
#   ※ 尾の表は実験117 の材料（実験だけに使う材料を含む）から作っている。苗として配るときは、ライセンスの分かる材料で作り直す。
_src117 = open("experiments/exp117_ending_swap.py", encoding="utf-8").read().split('res = dict(exp="117")')[0]
_ns117 = {}
exec(_src117, _ns117)
swap, split117, peel, polite_tail = _ns117["swap"], _ns117["split"], _ns117["peel"], _ns117["polite"]
FUNC117 = _ns117["FUNC"]
EMO = {"感動", "嬉しい", "楽しい", "悲しい", "辛い", "大変", "疲れる", "驚く", "困る", "怖い", "寂しい", "面白い", "美味しい", "好き", "嫌い", "心配", "残念", "最高", "幸せ", "ショック", "緊張"}


isq = lambda t: t.rstrip().endswith(("？", "?", "か", "か。", "の？"))


def morphs(text):
    return [(m.surface(), m.normalized_form(), "-".join(x for x in m.part_of_speech() if x != "*")) for m in tok.tokenize(text)]


def last_predicate(ms):
    """文の最後の述語の文節（感情の語があればそれを優先）。"""
    idx = [i for i, w in enumerate(ms) if not w[2].startswith(FUNC117) and ("非自立可能" in w[2] or "非自立" not in w[2]) and w[2].startswith(("動詞", "形容詞", "形状詞", "名詞"))
           and not (i > 0 and ms[i - 1][2].startswith("助詞-接続助詞"))]
    emo = [i for i in idx if ms[i][1] in EMO]
    return (emo or idx or [None])[-1]


def backtrack(user):
    """「昨日、映画を見てすごく感動したよ」→「感動したんですね」。相手の述語（感情の語を優先）を、くだけた形 + んですね で返す。質問には返さない。"""
    if isq(user):
        return ""
    ms = morphs(user); k = last_predicate(ms)
    if k is None:
        return ""
    j = k + 1
    while j < len(ms) and (ms[j][2].startswith(FUNC117) or "非自立" in ms[j][2]) and not ms[j][2].startswith(("助詞-接続助詞", "補助記号")):
        j += 1
    core, tail = split117(ms[k:j]); tail, _ = peel(tail)
    tail = [w for w in tail if not w[2].startswith("助詞-終助詞")]
    while tail and tail[-1][1] in ("だ", "です", "の", "ん") and not (tail[-1][1] == "だ" and tail[-1][0] in ("だっ",)):
        tail = tail[:-1]                                                                       # 〜んだ・〜のだ・〜で（説明・つなぎ）は外して「んですね」に付け替える
    lem = ms[k][1]; pos = ms[k][2]
    arg = argument(ms, k)                                                                      # 持ち主の案: 「何に・何が」どうした
    if pos.startswith(("名詞", "形状詞")) and not [w for w in tail if w[1] not in ("だ", "で")]:
        return arg + "".join(w[0] for w in core) + "なんですね。"                             # 散歩が大変で → 散歩が大変なんですね
    if not tail:
        return arg + "".join(w[0] for w in core) + "んですね。"
    if polite_tail(tail):
        past = any(w[1] == "た" for w in tail); neg = any(w[1] in ("ない", "ん", "ぬ") for w in tail)
        out, _ = swap(core, tail, False)
        if out and past and out.endswith(("た", "だ")) and "ください" not in out:
            return arg + out + "んですね。"
        if not past and not neg:
            return arg + lem + "んですね。"                                                    # 北海道に行きます → 北海道に行くんですね（辞書の形）
        return ""
    return arg + "".join(w[0] for w in core + tail) + "んですね。"


def argument(ms, k):
    """述語のすぐ前の「名詞 + が・を・に・へ・で・と」（何に・何が）。は（主題）や 私・僕 は付けない。"""
    if k >= 2 and ms[k - 1][2].startswith("助詞-格助詞") and ms[k - 1][0] in ("が", "を", "に", "へ", "で", "と"):
        j = k - 2
        while j >= 0 and ms[j][2].startswith(("名詞", "接頭辞", "接尾辞")) and "副詞可能" not in ms[j][2]:   # 最近・今日 などは付けない
            j -= 1
        noun = "".join(w[0] for w in ms[j + 1:k - 1])
        if noun and ms[k - 2][1] not in ("私", "僕", "俺", "自分", "何", "事", "物"):
            return noun + ms[k - 1][0]
    return ""


def to_polite(text):
    """返事の最後の文の結びを丁寧に（基本は丁寧、持ち主 2026-11-02）。"""
    ms = morphs(text); k = last_predicate(ms)
    if k is None:
        return text
    core, tail = split117(ms[k:]); tail, fin = peel(tail)
    if not tail or polite_tail(tail) or text.rstrip().endswith(("？", "?")):
        return text
    out, _ = swap(core, tail, True)
    return "".join(w[0] for w in ms[:k]) + out + "".join(w[0] for w in fin) if out else text


# ---- 話題を続ける質問（持ち主の案「北海道 → 札幌ですか？」）: 同じ発言の中で一緒に出やすい名詞（属性先の語板の代わり、数え上げ）----
CO = defaultdict(Counter); NOUNF = Counter()
for t in POOL[:150000]:
    ns = {m.normalized_form() for m in tok.tokenize(t) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("固有名詞", "普通名詞") and len(m.surface()) >= 2}
    for a in ns:
        NOUNF[a] += 1
        for b in ns:
            if a != b:
                CO[a][b] += 1
print("名詞の一緒に出る表", len(CO), flush=True)


def related_question(user, history):
    """相手の名詞（固有名詞を優先）から、一緒に出やすい名詞を選んで「〜ですか？」と聞く。"""
    nouns = [m for m in tok.tokenize(user) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("固有名詞", "普通名詞") and len(m.surface()) >= 2]
    nouns.sort(key=lambda m: (m.part_of_speech()[1] != "固有名詞", NOUNF[m.normalized_form()]))
    said = "".join(history)
    for m in nouns:
        x = m.normalized_form(); c = CO.get(x)
        if not c or NOUNF[x] < 5:
            continue
        best = max(((v / math.sqrt(NOUNF[y] + 10), y) for y, v in c.items() if v >= 3 and y not in said and NOUNF[y] >= 5 and len(y) >= 2 and y not in STOP and not all("ぁ" <= ch <= "ゖ" for ch in y)), default=None)   # ひらがなだけの語（こと・もの）は外す
        if best:
            return f"{best[1]}ですか？"
    return None


SESSION_NEW = {}                                                                               # 聞いた話から作る語（バッファ）: 語 → 一緒に聞いた語


def unknown_word(user):
    """知らない語（教えの会話に出ない名詞）があれば聞き返す。聞いた話はバッファに残す（持ち主の案: 語板をリアルタイムに増やす）。"""
    for m in tok.tokenize(user):
        if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("固有名詞", "普通名詞") and len(m.surface()) >= 2:
            x = m.normalized_form()
            if cw.get(x, 0) + cnext.get(x, 0) == 0 and x not in SESSION_NEW:
                SESSION_NEW[x] = set(analyze(user)[0]) - {x}
                return m.surface()
    return None


def reply(history):
    """感情（「何に」どうした、のバックトラッキング）→ 内容（揺らした 10 通りの票で選ぶ）→ 質問（話題を続ける）。知らない語は聞き返す。"""
    bt = backtrack(history[-1])
    new = unknown_word(history[-1])
    if new:
        return bt + f"「{new}」って何ですか？ 初めて聞きました。", 0.0
    scored = candidates(history)
    if not scored:
        return bt + "もう少し教えてもらえますか？", 0.0
    best, conf = ensemble(scored)
    parts = [bt]
    if conf < 0.5:                                                                             # 票が割れた = 自信がない → 内容は出さず、話題を続ける質問
        rq = related_question(history[-1], history)
        if rq:
            return bt + rq, conf
        parts.append("（自信がないけど）")
    parts.append(to_polite(best))
    if not isq(best) and len(best) < 30:                                                        # 長い返事には質問を足さない
        rq = related_question(history[-1], history + [best])
        if rq:
            parts.append(rq)
    return "".join(p for p in parts if p), conf


tests = ARGS or ["昨日、映画を見てすごく感動したよ", "こんにちは！最近ラーメンにはまってるんだ", "休みの日は何をしていますか？", "犬を飼っているんですが、散歩が大変で", "明日から旅行で北海道に行きます", "最近寝不足でつらいよ", "週末にポポロンタをやったよ"]
out = []
for u in tests:
    r, sc = reply([u]); out.append(dict(user=u, bot=r, confidence=round(sc, 2)))
    print("人:", u); print("ボット:", r, f"（自信 {sc:.1f}）"); print()
json.dump(out, open("results/testbot02_examples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
