"""テストボット02: 返事を選ぶ会話ボット（実験115 の選び方 C: 語の重なり + 次の発言の共鳴）+ 口調の合わせ（丁寧・くだけた）。
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


def reply(history):
    uw, up = analyze(history[-1])
    keys = set(uw) | {y for x in uw for y in NEXT.get(x, [])[:40]}
    cand = Counter()
    for y in keys:
        for k in INV.get(y, [])[:3000]:
            cand[k] += 1
    ids = [k for k, _ in cand.most_common(400)]
    if not ids:
        return "うーん、よく分からないです。もう少し教えてもらえますか？", 0.0
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
    best = scored[0]
    if best[0] < 0.8:                                                                           # 手がかりが弱い = 自信がない
        return f"（自信がないけど）{best[1]}", best[0]
    return best[1], best[0]


tests = ARGS or ["こんにちは！最近ラーメンにはまってるんだ", "休みの日は何をしていますか？", "犬を飼っているんですが、散歩が大変で", "明日から旅行で北海道に行きます", "最近寝不足でつらいよ"]
out = []
for u in tests:
    r, sc = reply([u]); out.append(dict(user=u, bot=r, score=round(sc, 2)))
    print("人:", u); print("ボット:", r, f"（点 {sc:.2f}）"); print()
json.dump(out, open("results/testbot02_examples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
