"""実験70: 章をまたぐ行の連想 — 文章 1 単位（1 作品）の中で、ある 1 行が別の章のどの 1 行といちばん関係が深いかを、行板（語板の融合）で探せるか

着想: 2026-10-29 持ち主「文章 1 単位中において、ある 1 行が別の章のある 1 行に最も関係がある、深い、ということを検索というか、連想できるかの実験も」。
方法: 青空文庫の作品（著作権切れ、非公開フォルダの本文から読む。文そのものは結果に出さない）を 1 作品 = 文章 1 単位とし、前から 4 等分して章にする。
      行 = 句点（。）で区切った文のうち、語板を持つ名詞が 2 つ以上あるもの。
      行板 = その行の名詞の語板（実験59、名詞 2 万語）の出口の明るさの平均（学び 34・65: 明るさで足す融合）= 5 観点 × 52 値の明るさの模様。
      「関係が深い」の正解: その作品の中で、ちょうど 2 つの行（しかも別の章）にだけ出てくる名詞（めったに出ない物・人・場所）でつながった 2 行。
      その名詞そのものは両方の行から取り除く（そのまま同じ語を探すだけになるのを防ぐ）。残りの語から連想して相手の行を当てられるか。
  探し方:
    行板の重なり（明るさの模様の cos 類似度）。光では、相手の行板を透過の模様（フィルム）にして問いの行板の光を通し、通り抜けた光の量を測る（干渉しない相関器）
    行板（めったに出ない語を重く）: 作品の中で出る行の少ない語ほど重く融合（明るい光で入れる）
    比べ: 同じ名詞の重なり（取り除いた名詞以外で、名詞がいくつ共通か）/ 2 つの順位を足す / でたらめ
  指標: 別の章の全部の行の中で、正解の行が 1 位・上位 5 に入る割合、順位の逆数の平均（MRR）。
結果の置き場: results/exp70.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp70_cross_chapter_link.py
"""
import collections, json, math, os, random, re, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t00 = time.time()
M = torch.load("D:/AI_git/Sin-CNN3_private/models/goban_名詞_seed0.pt", weights_only=False)      # 実験59 の名詞の語板（2 万語、全部の材料）
WORDS = M["words"]; WI = {w: i for i, w in enumerate(WORDS)}
with torch.no_grad():
    B = torch.cat([run(M["P"][i:i + 5000], 5, 52) for i in range(0, len(WORDS), 5000)])     # 全部の語板に光を通した出口の明るさ (語, 観点, 値)
B = B / B.sum(-1, keepdim=True).clamp(min=1e-12)                                           # 観点ごとに割合に直す
B = B.reshape(len(WORDS), -1)                                                              # 1 語 = 260 個の明るさの模様
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").create(); MODE = SplitMode.C
docs = [json.loads(l) for l in open("D:/AI_git/Sin-CNN3_private/data/text/aozora_full.jsonl", encoding="utf-8")]
rnd = random.Random(0); docs = [d for d in docs if 30000 < len(d["text"]) < 300000]; rnd.shuffle(docs); docs = docs[:60]
print("作品", len(docs), f"{time.time() - t00:.0f} 秒", flush=True)
hits = collections.defaultdict(lambda: [0, 0, 0.0]); n_q = 0; per_doc = []
for d in docs:
    sents = [s for s in re.split(r"(?<=。)", d["text"].replace("\n", "")) if len(s) > 8]
    lines = []
    for k, s in enumerate(sents):
        ns = sorted({m.normalized_form() for m in tok.tokenize(s[:500], MODE) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "普通名詞" and m.normalized_form() in WI})
        if len(ns) >= 2:
            lines.append((k * 4 // len(sents), ns))                                         # (章 0〜3, 名詞)
    if len(lines) < 40:
        continue
    occ = collections.defaultdict(list)
    for li, (ch, ns) in enumerate(lines):
        for n in ns:
            occ[n].append(li)
    pairs = [(n, ls[0], ls[1]) for n, ls in occ.items() if len(ls) == 2 and lines[ls[0]][0] != lines[ls[1]][0]]
    if not pairs:
        continue
    for n, a, b in pairs[:20]:
        def board(li, idf=False):
            ws = [w for w in lines[li][1] if w != n]
            if not ws:
                return None
            wt = torch.tensor([math.log(len(lines) / len(occ[w])) if idf else 1.0 for w in ws])   # idf: 作品の中でめったに出ない語ほど重く（明るい光で融合）
            return (B[[WI[w] for w in ws]] * wt[:, None]).sum(0) / wt.sum().clamp(min=1e-9)
        qa = board(a)
        if qa is None:
            continue
        cands = [li for li, (ch, ns) in enumerate(lines) if ch != lines[a][0]]               # 別の章の全部の行
        boards = [board(li) for li in cands]
        keep = [(li, bd) for li, bd in zip(cands, boards) if bd is not None]
        if len(keep) < 5 or b not in [li for li, _ in keep]:
            continue
        ids = [li for li, _ in keep]; Mx = torch.stack([bd for _, bd in keep])
        sc_goban = torch.nn.functional.cosine_similarity(Mx, qa[None], dim=1)               # 行板の明るさの模様の重なり
        qi = board(a, idf=True); Mi = torch.stack([board(li, idf=True) for li in ids])
        sc_idf = torch.nn.functional.cosine_similarity(Mi, qi[None], dim=1)                # めったに出ない語を重くした行板
        qset = set(lines[a][1]) - {n}
        sc_bow = torch.tensor([len(qset & (set(lines[li][1]) - {n})) + 1e-3 * rnd.random() for li in ids])   # 共通の名詞の数（同点はでたらめに）
        sc_rand = torch.rand(len(ids), generator=torch.Generator().manual_seed(n_q))
        pos_b = ids.index(b); n_q += 1
        rk = lambda x: torch.argsort(torch.argsort(-x)).float()                           # 順位（0 が 1 位）
        sc_both = -(rk(sc_idf) + rk(sc_bow))                                                # 2 つの順位を足して合わせる
        for name, sc in (("行板の重なり（語板の融合）", sc_goban), ("行板（めったに出ない語を重く）", sc_idf), ("同じ名詞の数", sc_bow), ("行板（重く）+ 同じ名詞の数", sc_both), ("でたらめ", sc_rand)):
            r = int((sc > sc[pos_b]).sum())
            hits[name][0] += r < 1; hits[name][1] += r < 5; hits[name][2] += 1 / (r + 1)
        hits["候補の数"][2] += len(ids)
    per_doc.append(len(pairs))
res = {k: dict(top1=v[0] / n_q, top5=v[1] / n_q, MRR=v[2] / n_q) for k, v in hits.items() if k != "候補の数"}
res["問題の数"] = n_q; res["候補の数の平均"] = hits["候補の数"][2] / max(n_q, 1); res["作品"] = len(per_doc); res["sec"] = time.time() - t00
print(json.dumps(res, ensure_ascii=False, indent=1))
with open("results/exp70.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(exp="70", **res), ensure_ascii=False) + "\n")
