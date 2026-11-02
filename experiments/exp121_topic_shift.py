"""実験121: 会話の方向が変わったことを、前の話題（属性）とのずれで見つける（持ち主の案 2026-11-02「以前の会話の属性との不一致が大きいなら、新しい会話が始まったとして自信の票もリセット」）

ずれ = 1 − つながり。つながり = 今の発言の語と、前の話題（前の 4 発言の語、古いほど 0.7 倍）の 重なり + 次の発言の共鳴（PMI）。
話題を変える言葉（ところで・そういえば・話は変わるけど…）があれば、ずれを大きくする。
試し（答えは作り方から分かる）: 試しの会話 A の途中 4 発言のあとに、別の会話 B の途中の発言をつなぐ。
  「B の発言（切り替わり）」と「A の続きの発言（切り替わりでない）」を、ずれで見分けられるか（AUC と、しきい値ごとの正答）。
材料: RealPersonaChat（nu-dialogue、CC BY-SA 4.0）。結果: results/exp121.txt、docs/testbot02.md に追記。実行: python experiments/exp121_topic_shift.py
"""
import json, math, random, sys
from collections import Counter

src = open("experiments/exp115_response_selection.py", encoding="utf-8").read().split("# ---- 試し: 10 択 ----")[0]
exec(src)                                                                                      # 材料・解析・共鳴（実験115）
cache = {}
an = lambda t: cache.setdefault(t, analyze(t))
SHIFT_WORDS = ("ところで", "そういえば", "話は変わ", "話変わ", "全然関係ない", "別の話")


def connection(ctx, u):
    ctxw = Counter()
    for k, c in enumerate(reversed(ctx)):
        for w in an(c)[0]:
            ctxw[w] += 0.7 ** k
    uw = an(u)[0]
    if not uw or not ctxw:
        return 0.0
    over = sum(v for w, v in ctxw.items() if w in uw) / math.sqrt(len(uw))
    res = sum(v * pmi(x, y) for x, v in ctxw.items() for y in uw) / math.sqrt(len(uw) * sum(ctxw.values()))
    return over + res


def shift(ctx, u):
    s = -connection(ctx, u)
    if any(w in u for w in SHIFT_WORDS):
        s += 5.0
    return s


rng = random.Random(0)
rpc = [t for s, t in convs["test"] if s == "real-persona-chat" and len(t) >= 10]
pos, neg = [], []
for turns in rpc:
    i = rng.randrange(4, len(turns) - 1); ctx = turns[i - 4:i]
    other = rng.choice(rpc); j = rng.randrange(4, len(other))
    pos.append(shift(ctx, other[j])); neg.append(shift(ctx, turns[i]))                         # pos = 切り替わり（別の会話の発言）、neg = 続き
auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in rng.sample(neg, min(200, len(neg)))) / (len(pos) * min(200, len(neg)))
res = dict(exp="121", n=len(pos), auc=auc, thresholds={})
for th in (-0.5, -1.0, -2.0, -3.0):
    tp = sum(p > th for p in pos) / len(pos); tn = sum(n <= th for n in neg) / len(neg)
    res["thresholds"][th] = dict(切り替わりを見つける=round(tp, 3), 続きを続きとみなす=round(tn, 3))
print("切り替わりの見分け AUC", round(auc, 3), "問", len(pos), res["thresholds"], flush=True)
with open("results/exp121.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
