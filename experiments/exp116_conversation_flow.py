"""実験116: 会話の流れ（今どこを話しているか）を持たせる。実験115 の「同じ会話の中から選ぶ」38% を上げる（苗の最低限の形、持ち主 2026-11-02）

実験115 で分かったこと: 文脈を 3 発言に広げると、同じ会話の前の話題に引っ張られて悪くなる（38% → 29%）。
流れの手がかり（どれも中身の語・文の形だけ、言語の知識は少し）:
  直前の発言との 重なり・共鳴 / それより前の発言との 重なり・共鳴（古い話題。足すか引くかは学ぶ）/
  もう言われた語の割合（同じことの繰り返しを避ける）/ 直前が質問（？・か）なら、候補が答えの形か（質問で返していないか）/
  候補が質問か / 長さの比 / 口調の合い / 候補が会話の何番目に来やすい形か（あいさつ・お礼は始めと終わり）。
これらの重みを、教えの会話の「正しい次の発言 vs 同じ会話のほかの発言」でロジスティック回帰で学ぶ（答えは本物の会話そのもの、ほかの AI は使わない）。
比べ: 実験115 の 重なり + 共鳴（同じ会話から 38%、ほかの会話から 61%）。
結果: results/exp116.txt、docs/testbot02.md に追記、図 docs/figs/conversation_flow_116.svg。実行: python experiments/exp116_conversation_flow.py
"""
import json, math, random, sys
from collections import Counter

import torch

src = open("experiments/exp115_response_selection.py", encoding="utf-8").read().split("# ---- 試し: 10 択 ----")[0]
exec(src)                                                                                      # 材料・解析・次の発言の共鳴（実験115）
torch.manual_seed(0)
cache = {}
an = lambda t: cache.setdefault(t, analyze(t))
isq = lambda t: t.rstrip().endswith(("？", "?", "か", "か。", "かね", "の？"))
GREET = ("こんにちは", "こんばんは", "おはよう", "よろしく", "ありがとう", "失礼", "さようなら", "またね", "お疲れ")
FEATS = ["重なり 直前", "共鳴 直前", "重なり 前の話題", "共鳴 前の話題", "もう言われた語の割合", "直前が質問 × 候補が質問でない", "候補が質問", "長さの比", "口調の合い", "あいさつ × 会話の始め・終わり", "あいさつ × 会話の途中", "重なり 自分の前の発言", "共鳴 自分の前の発言", "文の形の近さ 自分の前の発言"]


def resonance(uw, cw_):
    return sum(pmi(x, y) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_)) if uw and cw_ else 0.0


def feats(ctx, cand, pos, n):
    cw_, cp = an(cand); lw, lp = an(ctx[-1])
    older = set().union(*[an(u)[0] for u in ctx[:-1]]) if len(ctx) > 1 else set()
    said = set().union(*[an(u)[0] for u in ctx])
    g = any(cand.startswith(x) or x in cand[:12] for x in GREET); edge = pos <= 1 or pos >= n - 2
    return [len(lw & cw_) / math.sqrt(len(cw_) + 1), resonance(lw, cw_), len(older & cw_) / math.sqrt(len(cw_) + 1), resonance(older, cw_),
            len(said & cw_) / (len(cw_) + 1), float(isq(ctx[-1]) and not isq(cand)), float(isq(cand)),
            math.log((len(cand) + 1) / (len(ctx[-1]) + 1)), float(lp == cp), float(g and edge), float(g and not edge)] + own(ctx, cand)


def style(t):
    """文の形（言語の知識なし）: 文字の 2 つ組の袋（話し方のくせ: 語尾・記号・笑いの書き方 など）。"""
    return Counter(t[i:i + 2] for i in range(len(t) - 1))


def own(ctx, cand):
    """自分（次に話す人）の前の発言 = 2 つ前。同じ人の話し方・話題の続き。"""
    if len(ctx) < 2:
        return [0.0, 0.0, 0.0]
    mw, _ = an(ctx[-2]); cw_, _ = an(cand); a, b = style(ctx[-2]), style(cand)
    cos = sum(a[k] * b[k] for k in a) / math.sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()) + 1e-9)
    return [len(mw & cw_) / math.sqrt(len(cw_) + 1), resonance(mw, cw_), cos]


def make(data, per_conv, rng, k_neg=9):
    X, groups = [], []
    for src, turns in data:
        n = len(turns)
        for i in rng.sample(range(2, n), min(per_conv, n - 2)):
            ctx = turns[max(0, i - 4):i]
            same = [t for j, t in enumerate(turns) if abs(j - i) > 1 and t != turns[i]]
            if len(same) < k_neg:
                continue
            cands = [turns[i]] + rng.sample(same, k_neg)
            groups.append([feats(ctx, c, i, n) for c in cands])                                 # 0 番目が正解
    return torch.tensor(groups)


rng = random.Random(0)
TR = make(convs["train"][:6000], 3, rng); TE = make(convs["test"], 2, random.Random(1))
print("教えの問", len(TR), "試しの問", len(TE), flush=True)
mu, sd = TR.reshape(-1, len(FEATS)).mean(0), TR.reshape(-1, len(FEATS)).std(0) + 1e-6
w = torch.zeros(len(FEATS), requires_grad=True); opt = torch.optim.Adam([w], lr=0.05)
for step in range(400):
    logits = ((TR - mu) / sd) @ w                                                              # 10 個の候補の点
    loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(TR), dtype=torch.long)) + 1e-3 * w.pow(2).sum()
    opt.zero_grad(); loss.backward(); opt.step()


def top1(G, wv):
    sc = ((G - mu) / sd) @ wv
    return float((sc.argmax(1) == 0).float().mean())


base = torch.zeros(len(FEATS)); base[0] = 1 / sd[0] * sd[0]; base[1] = 1.0                  # 実験115 の 重なり + 共鳴（直前だけ）に近い形
b115 = float(((TE[:, :, 0] + TE[:, :, 1]).argmax(1) == 0).float().mean())
res = dict(exp="116", n_train=len(TR), n_test=len(TE), baseline_115=b115, learned=top1(TE, w.detach()),
           weights={f: round(float(x), 3) for f, x in zip(FEATS, w.detach())})
# 1 つずつ外して、どの手がかりが効いているか
res["drop_one"] = {}
for i, f in enumerate(FEATS):
    w2 = w.detach().clone(); w2[i] = 0; res["drop_one"][f] = round(res["learned"] - top1(TE, w2), 4)
print("実験115 の形（同じ会話から）", round(b115, 3), "流れの手がかりを学ぶ", round(res["learned"], 3), flush=True)
print("重み", res["weights"], flush=True); print("外すと下がる", res["drop_one"], flush=True)
with open("results/exp116.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
