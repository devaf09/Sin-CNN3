"""実験80 の「値の仲間 45」（観点「を」の値 2,000 を、名詞との組み合わせの似かたで分けたもの）を全部出す（持ち主の依頼 2026-10-31）。
分け方は experiments/exp80_attribute_boards.py と同じ（同じ材料・同じ乱数なので同じ仲間になる）。各仲間の値を、よく出る順に並べる。
出力: docs/value_clusters_80.md。実行: python tools/value_clusters_80.py"""
import collections, pickle, sys

import torch

sys.stdout.reconfigure(encoding="utf-8")
PRIV = "D:/AI_git/Sin-CNN3_private"; VIEW = "を"; NVAL, NCL, CAP = 2000, 45, 240
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
D = pickle.load(open(f"{PRIV}/data/views/merged2.pkl", "rb"))
ALL = [n for n in sorted(D["noun_total"], key=D["noun_total"].get, reverse=True) if n not in SKIP][:30000]
WORDS = ALL[::5]
tot = collections.Counter()
for n in ALL:
    tot.update(D["train"][n].get(VIEW, {}))
VALS = [x for x, _ in tot.most_common(NVAL)]; VI = {x: i for i, x in enumerate(VALS)}
TR = torch.zeros(len(WORDS), NVAL + 1)
for i, n in enumerate(WORDS):
    for x, m in D["train"].get(n, {}).get(VIEW, {}).items():
        TR[i, VI.get(x, NVAL)] += m
Xv = torch.nn.functional.normalize(TR[:, :NVAL].T.sqrt(), dim=1)
g = torch.Generator().manual_seed(0); C = Xv[torch.randperm(NVAL, generator=g)[:NCL]].clone()
for it in range(25):
    sim = Xv @ C.T; lab = torch.full((NVAL,), -1, dtype=torch.long); cnt = torch.zeros(NCL, dtype=torch.long)
    for v in torch.argsort(sim.max(1).values, descending=True).tolist():
        for c in torch.argsort(sim[v], descending=True).tolist():
            if cnt[c] < CAP:
                lab[v] = c; cnt[c] += 1; break
    for c in range(NCL):
        if (lab == c).any():
            C[c] = torch.nn.functional.normalize(Xv[lab == c].mean(0), dim=0)
# 仲間ごとに、その仲間と組むことの多い名詞（「を」の相手）も出す: 仲間の値への光の合計が多い名詞
share = TR[:, :NVAL] / TR[:, :NVAL].sum(1, keepdim=True).clamp(min=1)
L = ["# 値の仲間 45（実験80、観点「を」の値 2,000）", "",
     "観点「を」（何を〜する）の値 = 動詞 2,000 個を、「どの名詞と組むか」の似かたで 45 の仲間に分けた（k 平均法、1 つの仲間は 240 まで）。",
     "教えたのは「名詞 を 動詞」の数だけで、意味は教えていない。それでも仲間は意味のまとまりになった。各仲間: 値（よく出る順に 20 まで）と、その仲間の動詞を「を」でよく取る名詞（5 つ）。", "",
     "| 仲間 | 数 | 値（動詞、よく出る順） | よく組む名詞 |", "|---|---|---|---|"]
order = sorted(range(NCL), key=lambda c: -sum(tot[VALS[int(v)]] for v in torch.nonzero(lab == c).flatten()))
for c in order:
    mem = sorted([VALS[int(v)] for v in torch.nonzero(lab == c).flatten()], key=lambda x: -tot[x])
    ids = torch.nonzero(lab == c).flatten()
    nscore = share[:, ids].sum(1) * TR[:, ids].sum(1).clamp(min=1).log()
    top_n = [WORDS[int(i)] for i in torch.argsort(nscore, descending=True)[:5]]
    L.append(f"| {c} | {len(mem)} | {'・'.join(mem[:20])}{' …' if len(mem) > 20 else ''} | {'・'.join(top_n)} |")
open("docs/value_clusters_80.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[6:]))
