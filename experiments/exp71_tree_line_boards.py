"""実験71: 行板を属性樹（知性樹）で補強する — 章をまたぐ行の連想（実験70）は良くなるか

着想: 2026-10-29 持ち主「実験70 の連想の方法は的確。次は行板を属性樹で補強できないか考えたい」。
      実験70 の行板（行の名詞の語板の明るさの平均）は、行の名詞が 2〜5 個と少なく、同じ仲間の別の語（ラーメンとうどん）では行どうしがつながりにくい。
      属性樹（実験65）の節は「使われ方が似た語の仲間」なので、その仲間の情報を足せば、連想が広がるはず。
方法: 実験59 の名詞の語板 2 万語で属性樹を作る（実験65 と同じ考え: 共通の度合いの大きい属性で枝分かれ、深さ 4、1 つの節から最大 6 本）。
      節の仲間 = 根からその節までの属性を全部持つ語。節の融合した板 = 仲間の語板の明るさの平均。
  1. 木でならした語板: 語板 = (1 − α) × 自分の語板 + α × いちばん深い節の融合した板（α = 0.3・0.5）
  2. 木の道筋の目印: 行の名詞が通る節に印を付け（深い節ほど重く = 2^深さ）、行どうしで印の重なり（cos）を測る
  3. 組み合わせ: 行板（ならした）+ 道筋の目印 + 共通の名詞の数 の順位を足す
  問題は実験70 と同じ（青空文庫 60 作品、4 章、2 行にだけ出る名詞でつながる組、その名詞は取り除く、1200 問）。
結果の置き場: results/exp71.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp71_tree_line_boards.py
"""
import collections, json, math, os, random, re, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t00 = time.time()
# MODEL=v1（実験59、観点 5）/ v2（実験73、観点 9）。COMMON=1 なら、両方にある語だけで問題を作る（v1 と v2 を同じ問題で比べる、2026-10-30）
MODEL = os.environ.get("MODEL", "v1"); COMMON = os.environ.get("COMMON") == "1"
M = torch.load("D:/AI_git/Sin-CNN3_private/models/goban_名詞_seed0.pt", weights_only=False)
if MODEL == "v2":
    P = []; WORDS = []
    for k in (0, 1):                                                                        # 語板 v2 は名詞を 2 つに分けて学んだので、合わせる
        M2 = torch.load(f"D:/AI_git/Sin-CNN3_private/models/goban_v2_part{k}.pt", weights_only=False); P.append(M2["P"]); WORDS += M2["words"]
    NVIEW, NVAL = 9, 45; P = torch.cat(P)
else:
    WORDS = M["words"]; NVIEW, NVAL = 5, 47; P = M["P"]
WI = {w: i for i, w in enumerate(WORDS)}
QWI = WI
if COMMON:
    other_words = set(M["words"]) if MODEL == "v2" else set(w for k in (0, 1) for w in torch.load(f"D:/AI_git/Sin-CNN3_private/models/goban_v2_part{k}.pt", weights_only=False)["words"])
    QWI = {w: i for w, i in WI.items() if w in other_words}                                 # 問題に使う語（両方の語板にある語）
with torch.no_grad():
    B3 = torch.cat([run(P[i:i + 5000], NVIEW, NVAL + NVIEW) for i in range(0, len(WORDS), 5000)])
B3 = B3 / B3.sum(-1, keepdim=True).clamp(min=1e-12)                                         # (語, 観点, 値) の割合
B = B3.reshape(len(WORDS), -1)                                                              # 1 語 = 260 個の明るさの模様
# ---- 属性樹（実験65 の考え、2 万語） ----
ATTR = [(v, x) for v in range(NVIEW) for x in range(NVAL)]                                        # 属性 = (観点, 値)（その他は除く）
prior = B3.mean(0)
has = torch.stack([B3[:, v, x] >= max(0.05, 3 * float(prior[v, x])) for v, x in ATTR], 1)    # (語, 属性) 持つか
nodes = [dict(depth=0, members=torch.arange(len(WORDS)), path=[])]                         # 根 = 全部の語


def grow(i, used):
    nd = nodes[i]
    if nd["depth"] >= 4 or len(nd["members"]) < 20:
        return
    share = has[nd["members"]].float().mean(0)
    for a in used:
        share[a] = -1
    for a in torch.argsort(share, descending=True)[:6].tolist():                            # 共通の度合いの大きい属性 6 つで枝
        if share[a] < 0.05:
            break
        mem = nd["members"][has[nd["members"], a]]
        if len(mem) < 10 or len(mem) == len(nd["members"]):
            continue
        nodes.append(dict(depth=nd["depth"] + 1, members=mem, path=nd["path"] + [a])); grow(len(nodes) - 1, used + [a])


grow(0, [])
fused = torch.stack([B[nd["members"]].mean(0) for nd in nodes])                            # 節の融合した板
deep = torch.zeros(len(WORDS), dtype=torch.long); deep_d = torch.zeros(len(WORDS), dtype=torch.long)
tag = torch.zeros(len(WORDS), len(nodes))                                                   # 語が通る節の印（深いほど重く）
for j, nd in enumerate(nodes):
    m = nd["members"]; tag[m, j] = 2.0 ** nd["depth"]
    better = nd["depth"] > deep_d[m]; deep[m[better]] = j; deep_d[m[better]] = nd["depth"]
print("属性樹の節", len(nodes), "深さの分布", collections.Counter(deep_d.tolist()), f"{time.time() - t00:.0f} 秒", flush=True)
smooth = {a: (1 - a) * B + a * fused[deep] for a in (0.3, 0.5)}                              # 木でならした語板
# ---- 問題（実験70 と同じ） ----
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").create(); MODE = SplitMode.C
docs = [json.loads(l) for l in open("D:/AI_git/Sin-CNN3_private/data/text/aozora_full.jsonl", encoding="utf-8")]
rnd = random.Random(0); docs = [d for d in docs if 30000 < len(d["text"]) < 300000]; rnd.shuffle(docs); docs = docs[:60]
hits = collections.defaultdict(lambda: [0, 0, 0.0]); n_q = 0
rk = lambda x: torch.argsort(torch.argsort(-x)).float()
for d in docs:
    sents = [s for s in re.split(r"(?<=。)", d["text"].replace("\n", "")) if len(s) > 8]
    lines = []
    for k, s in enumerate(sents):
        ns = sorted({m.normalized_form() for m in tok.tokenize(s[:500], MODE) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "普通名詞" and m.normalized_form() in QWI})
        if len(ns) >= 2:
            lines.append((k * 4 // len(sents), ns))
    if len(lines) < 40:
        continue
    occ = collections.defaultdict(list)
    for li, (ch, ns) in enumerate(lines):
        for n in ns:
            occ[n].append(li)
    pairs = [(n, ls[0], ls[1]) for n, ls in occ.items() if len(ls) == 2 and lines[ls[0]][0] != lines[ls[1]][0]]
    for n, a, b in pairs[:20]:
        ws = lambda li: [WI[w] for w in lines[li][1] if w != n]
        if not ws(a):
            continue
        ids = [li for li, (ch, _) in enumerate(lines) if ch != lines[a][0] and ws(li)]
        if len(ids) < 5 or b not in ids:
            continue
        pos_b = ids.index(b); n_q += 1
        board = lambda E, li: E[ws(li)].mean(0)
        sc = {}
        sc["行板（実験70）"] = torch.nn.functional.cosine_similarity(torch.stack([board(B, li) for li in ids]), board(B, a)[None], dim=1)
        for al, E in smooth.items():
            sc[f"木でならした行板 α={al}"] = torch.nn.functional.cosine_similarity(torch.stack([board(E, li) for li in ids]), board(E, a)[None], dim=1)
        sc["木の道筋の目印"] = torch.nn.functional.cosine_similarity(torch.stack([tag[ws(li)].sum(0) for li in ids]), tag[ws(a)].sum(0)[None], dim=1)
        qset = set(lines[a][1]) - {n}
        sc["共通の名詞の数"] = torch.tensor([len(qset & (set(lines[li][1]) - {n})) + 1e-3 * rnd.random() for li in ids])
        sc["行板 + 共通の名詞（実験70）"] = -(rk(sc["行板（実験70）"]) + rk(sc["共通の名詞の数"]))
        sc["ならした行板 + 道筋 + 共通の名詞"] = -(rk(sc["木でならした行板 α=0.3"]) + rk(sc["木の道筋の目印"]) + rk(sc["共通の名詞の数"]))
        sc["でたらめ"] = torch.rand(len(ids), generator=torch.Generator().manual_seed(n_q))
        for name, s in sc.items():
            r = int((s > s[pos_b]).sum()); hits[name][0] += r < 1; hits[name][1] += r < 5; hits[name][2] += 1 / (r + 1)
res = {k: dict(top1=v[0] / n_q, top5=v[1] / n_q, MRR=v[2] / n_q) for k, v in hits.items()}
res["問題の数"] = n_q; res["属性樹の節"] = len(nodes); res["sec"] = time.time() - t00
with open("results/exp71.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(exp="71", model=MODEL, common=COMMON, **res), ensure_ascii=False) + "\n")
for k, v in res.items():
    print(k, {a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else v)
