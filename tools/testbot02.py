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
        sc = len(uw & cw_) / math.sqrt(len(cw_) + 1)
        if uw and cw_:
            sc += sum(pmi(x, y) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_))
        sc += 0.3 if cp == up else 0.0                                                          # 口調の合い
        scored.append((sc, t))
    scored.sort(reverse=True)
    return scored


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
    if pos.startswith(("名詞", "形状詞")) and not [w for w in tail if w[1] not in ("だ", "で")]:
        return "".join(w[0] for w in core) + "なんですね。"                                   # 大変で → 大変なんですね
    if not tail:
        return "".join(w[0] for w in core) + "んですね。"
    if polite_tail(tail):
        past = any(w[1] == "た" for w in tail); neg = any(w[1] in ("ない", "ん", "ぬ") for w in tail)
        out, _ = swap(core, tail, False)
        if out and past and out.endswith(("た", "だ")) and "ください" not in out:
            return out + "んですね。"
        if not past and not neg:
            return lem + "んですね。"                                                          # 行きます → 行くんですね（辞書の形）
        return ""
    return "".join(w[0] for w in core + tail) + "んですね。"


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


def reply(history):
    """感情（バックトラッキング）→ 内容（共鳴で選んだ返事）→ 質問（上位に質問の候補があれば）。"""
    scored = candidates(history)
    if not scored:
        return "うーん、よく分からないです。もう少し教えてもらえますか？", 0.0
    best_sc, best = scored[0]
    parts = [backtrack(history[-1])]
    if best_sc < 0.8:
        parts.append("（自信がないけど）")                                                     # 手がかりが弱い = 自信がない
    parts.append(to_polite(best))
    if not isq(best) and len(best) < 30:                                                        # 長い返事には質問を足さない
        q = next((t for sc, t in scored[1:8] if isq(t) and sc >= 0.6 * best_sc and t not in history), None)
        if q:
            parts.append(to_polite(q))                                                         # 話をつなぐ質問
    return "".join(p for p in parts if p), best_sc


tests = ARGS or ["昨日、映画を見てすごく感動したよ", "こんにちは！最近ラーメンにはまってるんだ", "休みの日は何をしていますか？", "犬を飼っているんですが、散歩が大変で", "明日から旅行で北海道に行きます", "最近寝不足でつらいよ"]
out = []
for u in tests:
    r, sc = reply([u]); out.append(dict(user=u, bot=r, score=round(sc, 2)))
    print("人:", u); print("ボット:", r, f"（点 {sc:.2f}）"); print()
json.dump(out, open("results/testbot02_examples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
