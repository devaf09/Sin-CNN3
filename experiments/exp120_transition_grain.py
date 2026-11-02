"""実験120: 会話の状態遷移グレイン。連続する会話文から、ラベルなしで「発言の状態」を作り、状態 → 次の状態 を数える（持ち主 2026-11-02「連続する会話文から遷移グレインを作れれば」）

1. 発言の形の手がかり（中身の語ではなく話し方。言語の知識は少し）:
   終わりの 2 文字・終わりの記号（？！。…）、あいづち（そう・へー・なるほど・確かに）で始まるか、感情の語、
   自分の話（私・僕・自分）か、相手への問いかけ（〜さん・あなた・〜ですか）、前の発言との話題の重なり（続き / 切り替え）、長さ。
2. 手がかりの束を、ラベルなしで k 平均法で 16 の状態に分ける（実験82・91 と同じ、自然にできる仲間）。
3. 教えの会話で、状態 → 次の状態 の割合（遷移グレイン）を数える。
確かめ: 実験116 の「同じ会話の 10 択」（36.9%）で、点に log P(候補の状態 | 直前の状態) を足すと上がるか。
       状態ごとの中身を見て、質問・答え・共感・自分の話… と読めるか（名前は後から付ける）。
材料: RealPersonaChat・JMultiWOZ（nu-dialogue、CC BY-SA 4.0）。結果: results/exp120.txt、docs/testbot02.md に追記、図 docs/figs/transition_grain_120.svg。
実行: python experiments/exp120_transition_grain.py
"""
import json, math, random, sys
from collections import Counter, defaultdict

import torch

src = open("experiments/exp116_conversation_flow.py", encoding="utf-8").read().split("rng = random.Random(0)\nTR = make(")[0]
exec(src)                                                                                      # 材料・共鳴・流れの手がかり（実験115・116）
torch.manual_seed(0)
AIZ = ("そう", "へー", "へえ", "なるほど", "確かに", "たしかに", "いいですね", "わかります", "分かります", "ですよね", "ほんと", "本当")
EMO = ("嬉し", "楽し", "悲し", "辛", "つら", "大変", "疲れ", "驚", "困", "怖", "寂し", "面白", "美味し", "好き", "嫌", "心配", "残念", "最高", "幸せ", "羨まし", "すごい", "凄い")
SELF = ("私", "僕", "自分", "うち", "俺", "わたし")
YOU = ("さん", "あなた", "そちら")
ENDS = ["か？", "すか", "？", "ね！", "ね。", "ね", "よ！", "よ。", "よ", "す。", "す！", "た。", "！", "。", "w", "笑"]


def form(prev, t):
    """発言の形の手がかり（約 30 個）。"""
    tt = t.rstrip(); pw = an(prev)[0] if prev else set(); cw_ = an(t)[0]
    v = [float(tt.endswith(e)) for e in ENDS]
    v += [float(tt.startswith(AIZ)), float(any(e in tt for e in EMO)), float(any(tt.startswith(x) or (x + "は") in tt for x in SELF)),
          float(any(y in tt for y in YOU)), float(isq(tt)), float(bool(pw & cw_)), float(bool(pw) and not (pw & cw_)),
          min(len(tt), 80) / 80, float(any(g in tt[:12] for g in GREET)), float("？" in tt[:-1] or "?" in tt[:-1])]
    return v


rows, seqs = [], []
for src_, turns in convs["train"][:8000]:
    sq = []
    for i, t in enumerate(turns):
        rows.append(form(turns[i - 1] if i else "", t)); sq.append(len(rows) - 1)
    seqs.append(sq)
X = torch.tensor(rows)
mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd
K = 16
C = Z[torch.randperm(len(Z))[:K]].clone()
for it in range(25):                                                                           # k 平均法
    lab = torch.cdist(Z, C).argmin(1)
    for k in range(K):
        if (lab == k).any():
            C[k] = Z[lab == k].mean(0)
lab = torch.cdist(Z, C).argmin(1)
state = lambda prev, t: int(torch.cdist(((torch.tensor([form(prev, t)]) - mu) / sd), C).argmin(1))
T = torch.ones(K + 1, K)                                                                       # 遷移の数（始まり = K 番目の行）、1 でならす
for sq in seqs:
    prev = K
    for r in sq:
        T[prev, lab[r]] += 1; prev = int(lab[r])
logP = (T / T.sum(1, keepdim=True)).log()
# 状態の中身（読むため）: 手がかりの平均の目立つもの + 代表の発言（教えの会話）
names = ENDS + ["あいづちで始まる", "感情の語", "自分の話", "相手への問いかけ", "問い", "前と話題が続く", "前と話題が切り替わる", "長さ", "あいさつ", "途中に問い"]
desc = {}
flat = [(src_, t) for src_, turns in convs["train"][:8000] for t in turns]
for k in range(K):
    m = X[lab == k].mean(0); top = torch.argsort(m - X.mean(0), descending=True)[:3]
    ex = [flat[i][1] for i in (lab == k).nonzero().flatten()[:3].tolist()]
    desc[k] = dict(share=round(float((lab == k).float().mean()), 3), features=[names[i] for i in top.tolist()], examples=ex)
# 確かめ: 同じ会話の 10 択（実験116 と同じ作り）で、遷移の見込みを足す
rng = random.Random(1); ok0 = ok1 = ok2 = n = 0
for src_, turns in convs["test"]:
    for i in rng.sample(range(2, len(turns)), min(2, len(turns) - 2)):
        same = [t for j, t in enumerate(turns) if abs(j - i) > 1 and t != turns[i]]
        if len(same) < 9:
            continue
        cands = [turns[i]] + rng.sample(same, 9); ctx = turns[max(0, i - 4):i]
        s_prev = state(turns[i - 2] if i >= 2 else "", turns[i - 1])
        base = [feats(ctx, c, i, len(turns))[1] + feats(ctx, c, i, len(turns))[0] for c in cands]   # 実験115 の 重なり + 共鳴
        tr = [float(logP[s_prev, state(turns[i - 1], c)]) for c in cands]
        ok0 += max(range(10), key=lambda k: base[k]) == 0
        ok1 += max(range(10), key=lambda k: tr[k] + 1e-6 * rng.random()) == 0
        ok2 += max(range(10), key=lambda k: base[k] + 0.5 * tr[k]) == 0; n += 1
res = dict(exp="120", K=K, n=n, base=ok0 / n, transition_only=ok1 / n, base_plus_transition=ok2 / n, states=desc,
           top_transitions={k: [(int(j), round(float(v), 3)) for v, j in zip(*torch.topk((T[k] / T[k].sum()), 3))] for k in range(K)})
print("同じ会話の 10 択: 重なり + 共鳴", round(ok0 / n, 3), "遷移だけ", round(ok1 / n, 3), "重なり + 共鳴 + 遷移", round(ok2 / n, 3), "問", n, flush=True)
for k in range(K):
    print(k, desc[k]["share"], desc[k]["features"], "→", res["top_transitions"][k], flush=True)
with open("results/exp120.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
# テストボット02 で「次にどの状態で返すか」を決めるために、状態の分け方と遷移を保存する（非公開の置き場、材料は CC BY-SA の会話）
torch.save(dict(mu=mu, sd=sd, C=C, logP=logP, ENDS=ENDS, states=desc), "D:/AI_git/Sin-CNN3_private/models/transition_grain_120.pt")
