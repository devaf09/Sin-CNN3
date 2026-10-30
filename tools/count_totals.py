"""語板と属性樹の総量を数える（持ち主の依頼 2026-10-30）。非公開フォルダの models/ の語板ファイルと、語板 v3 の属性樹。実行: python tools/count_totals.py"""
import collections, glob, json, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4)
rows = []
for p in sorted(glob.glob("D:/AI_git/Sin-CNN3_private/models/goban_*.pt")):
    M = torch.load(p, weights_only=False); P = M["P"]
    rows.append(dict(file=os.path.basename(p), boards=P.shape[0], rows=P.shape[1], plates=P.shape[2], phases=int(P[0].numel()),
                     views=len(M.get("views") or M.get("vals") or []), mb=os.path.getsize(p) / 1e6))
# 語板 v3 の属性樹（実験71 と同じ作り方）
Ps = []; W = []
for k in (0, 1):
    M = torch.load(f"D:/AI_git/Sin-CNN3_private/models/goban_v3_part{k}_scratch.pt", weights_only=False); Ps.append(M["P"]); W += M["words"]
with torch.no_grad():
    P = torch.cat(Ps); B3 = torch.cat([run(P[i:i + 1500], 9, 54) for i in range(0, len(W), 1500)])
B3 = B3 / B3.sum(-1, keepdim=True).clamp(min=1e-12); FL = B3[:, :, :45].reshape(len(W), -1); prior = FL.mean(0)
has = FL >= torch.clamp(3 * prior, min=0.05)
nodes = [dict(depth=0, members=torch.arange(len(W)))]


def grow(i, used):
    nd = nodes[i]
    if nd["depth"] >= 4 or len(nd["members"]) < 20:
        return
    share = has[nd["members"]].float().mean(0)
    for u in used:
        share[u] = -1
    for a in torch.argsort(share, descending=True)[:6].tolist():
        if share[a] < 0.05:
            break
        mem = nd["members"][has[nd["members"], a]]
        if len(mem) < 10 or len(mem) == len(nd["members"]):
            continue
        nodes.append(dict(depth=nd["depth"] + 1, members=mem)); grow(len(nodes) - 1, used + [a])


grow(0, [])
deep = torch.zeros(len(W), dtype=torch.long)
for nd in nodes:
    deep[nd["members"]] = torch.maximum(deep[nd["members"]], torch.tensor(nd["depth"]))
tree = dict(nodes=len(nodes), by_depth=dict(collections.Counter(nd["depth"] for nd in nodes)), words_deepest=dict(collections.Counter(deep.tolist())),
            attrs_with_words=int(has.any(0).sum()), attrs_total=has.shape[1], mean_attrs_per_word=float(has.sum(1).float().mean()))
json.dump(dict(models=rows, tree_v3=tree), open("results/totals.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for r in rows:
    print(r)
print(tree)
