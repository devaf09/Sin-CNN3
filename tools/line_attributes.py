"""長めの行から 10 語ほどを、並びのまま取り出し、語ごとの属性の割合（足すと 1）を出す（持ち主の依頼 2026-10-31）。
語板 v3（名詞 3 万語、観点 9 × 値 45）の出口の明るさを、値の出口だけで全部足して 1 になるように割る（「その他」は除く）。
語板のない語（動詞・形容詞・語板のない名詞）は「語板なし」と出す。材料: 青空文庫（著作権切れ）。実行: python tools/line_attributes.py"""
import json, os, random, re, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4)
PRIV = "D:/AI_git/Sin-CNN3_private"
Ps = []; WORDS = []
for k in (0, 1):
    M = torch.load(f"{PRIV}/models/goban_v3_part{k}_scratch.pt", weights_only=False); Ps.append(M["P"]); WORDS += M["words"]
WI = {w: i for i, w in enumerate(WORDS)}; VIEWS = M["views"]
INV = {(j, r): x for j, v in enumerate(VIEWS) for x, r in M["pos"][v].items()}               # 出口の場所 → 値の名前
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").tokenizer(); MODE = SplitMode.C
SOURCE = os.environ.get("SOURCE", "aozora")                                                 # aozora（青空文庫）/ web2（持ち主が集めた Web 小説、実験だけ）
NEED = int(os.environ.get("N", "10"))                                                       # 出す行の数（傾向を数えるときは多めに）
if SOURCE == "web2":
    docs = [json.loads(l) for l in open(f"{PRIV}/data/text/webnovels2.jsonl", encoding="utf-8")]
else:
    docs = [json.loads(l) for l in open(f"{PRIV}/data/text/aozora_full.jsonl", encoding="utf-8")]
rnd = random.Random(31); rnd.shuffle(docs)
out = []
for d in docs:
    sents = [s for s in re.split(r"(?<=[。！？])", d["text"].replace("\r", "").replace("\n", "。" if SOURCE == "web2" else "").replace("　", "")) if 60 <= len(s) <= 120 and "底本" not in s and "---" not in s]
    if not sents:
        continue
    s = rnd.choice(sents)
    ms = [m for m in tok.tokenize(s, MODE) if m.part_of_speech()[0] in ("名詞", "動詞", "形容詞") and m.part_of_speech()[1] not in ("数詞", "非自立可能")]
    if len(ms) < 8:
        continue
    ws = [(m.surface(), m.normalized_form(), m.part_of_speech()[0]) for m in ms[:10]]       # 並びのまま 10 語まで
    ids = [WI[n] for _, n, _ in ws if n in WI]
    if len(ids) < 5:
        continue
    with torch.no_grad():
        W = run(torch.stack([Ps[0][0]] * 0 + [torch.cat(Ps)[i] for i in ids]), 9, 54)[:, :, :45]   # 値の出口だけ（「その他」は除く）
    W = W.reshape(len(ids), -1); W = W / W.sum(1, keepdim=True)                            # 1 語の属性の割合を足すと 1
    rows = []; k = 0
    for surf, n, pos in ws:
        if n in WI:
            p = W[k]; k += 1; top = torch.argsort(p, descending=True)[:5]
            rows.append((surf, n, [(f"{VIEWS[int(a) // 45]} {INV.get((int(a) // 45, int(a) % 45), '?')}", float(p[a])) for a in top]))
        else:
            rows.append((surf, n, None))
    out.append((d.get("title", ""), s, rows))
    if len(out) >= NEED:
        break
L = []
for i, (title, s, rows) in enumerate(out, 1):
    L.append(f"### {i}. {title}\n> {s}\n")
    for surf, n, top in rows:
        if top:
            L.append(f"- **{surf}**（{n}）: " + "、".join(f"{a} {p:.3f}" for a, p in top) + f"、ほか {1 - sum(p for _, p in top):.3f}")
        else:
            L.append(f"- {surf}（{n}）: 語板なし")
    L.append("")
# 傾向の数: 語ごとの上位 3 つの割合の和、上位 3 つに「は 無い・が 有る・どんな 無い」が入る割合、語板のある語の割合
GEN = {"は 無い", "が 有る", "どんな 無い"}
tops = [top for _, _, rows in out for _, _, top in rows if top]; allw = [r for _, _, rows in out for r in rows]
stat = dict(source=SOURCE, lines=len(out), words=len(allw), with_board=len(tops) / len(allw),
            top3_share=sum(sum(p for _, p in t[:3]) for t in tops) / len(tops),
            generic_in_top3=sum(any(a in GEN for a, _ in t[:3]) for t in tops) / len(tops),
            generic_top1=sum(t[0][0] in GEN for t in tops) / len(tops))
print(json.dumps(stat, ensure_ascii=False))
open(f"{PRIV}/docs/line_attributes_{SOURCE}.md", "w", encoding="utf-8").write("\n".join(L[:400]))   # 例は非公開フォルダへ（Web 小説は実験だけ）
if NEED <= 10:
    print("\n".join(L))
