"""実験118: 会話の始まり用に「話者に見えない会話」を頭に入れておく（持ち主の案 2026-11-02「最初の滑り出し用として」）

問題: 会話の始めは、話に出た語（キャッシュ）も直前の発言も少なく、返事を選ぶ手がかりが無い。
案: 見えない会話を、本当の会話の前に履歴として入れておく（相手には見せない）。返事の点を付けるときだけ、薄い重みで手がかりにする。
見えない会話の作り方（ほかの AI や手書きの文は使わない、材料から）:
  なし / よくある始まり: 教えの会話 200 本の「始めの 4 発言」をまとめた袋（会話の始まりの型）/
  誰かの会話: 教えの会話 1 本の発言（ボットの思い出・人格の種の代わり）。
物差し: 試しの会話の 1〜3 発言目で、正しい次の発言を 10 個の候補から選ぶ（でたらめ 10%）。
  外れの候補は ほかの会話の 1〜3 発言目（始まりどうしは似ていて、むずかしい）と、ほかの会話のどこか（実験115 と同じ）の 2 通り。
点: 実験115 の 重なり + 共鳴 を、履歴の発言ごとに重みを付けて足す（直前 1、その前 0.5、見えない会話は 0.3 × 0.97^古さ）。
材料: RealPersonaChat（nu-dialogue、CC BY-SA 4.0）。結果: results/exp118.txt、docs/testbot02.md に追記。実行: python experiments/exp118_hidden_primer.py
"""
import json, math, random, sys

src = open("experiments/exp115_response_selection.py", encoding="utf-8").read().split("# ---- 試し: 10 択 ----")[0]
exec(src)                                                                                      # 材料・解析・共鳴（実験115）
cache = {}
an = lambda t: cache.setdefault(t, analyze(t))
rpc_tr = [t for s, t in convs["train"] if s == "real-persona-chat"]; rpc_te = [t for s, t in convs["test"] if s == "real-persona-chat"]
rng = random.Random(0)
OPEN = [u for turns in rng.sample(rpc_tr, 200) for u in turns[:4]]                             # よくある始まりの袋


def item_score(u, cand):
    uw, _ = an(u); cw_, _ = an(cand)
    s = len(uw & cw_) / math.sqrt(len(cw_) + 1)
    if uw and cw_:
        s += sum(pmi(x, y) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_))
    return s


def score(hidden, ctx, cand, wh):
    sc = sum((1.0 if k == 0 else 0.5 ** k) * item_score(u, cand) for k, u in enumerate(reversed(ctx)))
    if hidden and wh > 0:
        sc += wh * sum(0.97 ** k * item_score(u, cand) for k, u in enumerate(reversed(hidden))) / math.sqrt(len(hidden))
    return sc


early_pool = [t for turns in rpc_te for t in turns[1:4]]; any_pool = [t for turns in rpc_te for t in turns]
cases = []
for ci, turns in enumerate(rpc_te):
    for i in (1, 2, 3):
        if i >= len(turns):
            continue
        negs_e = [t for t in rng.sample(early_pool, 30) if t not in turns][:9]; negs_a = [t for t in rng.sample(any_pool, 30) if t not in turns][:9]
        cases.append((i, turns[:i], turns[i], negs_e, negs_a))
cases = cases[:2400]
print("試しの問", len(cases), flush=True)
res = dict(exp="118", n=len(cases), runs=[])
for prim, hidden_fn in (("なし", lambda: None), ("よくある始まり", lambda: OPEN), ("誰かの会話", lambda: rng.choice(rpc_tr))):
    for wh in ((0.0,) if prim == "なし" else (0.3, 1.0)):
        for negkind in ("始まりどうし", "どこからでも"):
            ok = [0, 0, 0, 0]; n = [0, 0, 0, 0]
            for i, ctx, gold, ne, na in cases:
                hidden = hidden_fn(); cands = [gold] + (ne if negkind == "始まりどうし" else na)
                scs = [score(hidden, ctx, c, wh) + 1e-6 * rng.random() for c in cands]
                good = max(range(len(cands)), key=lambda k: scs[k]) == 0
                ok[i] += good; n[i] += 1
            r = dict(primer=prim, w=wh, neg=negkind, top1=sum(ok) / sum(n), by_turn={t: round(ok[t] / n[t], 3) for t in (1, 2, 3) if n[t]})
            res["runs"].append(r); print(prim, wh, negkind, "1 位", round(r["top1"], 3), r["by_turn"], flush=True)
with open("results/exp118.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
