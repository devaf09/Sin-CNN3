"""実験88: 文章樹で (a) 属性から人物を特定する流れ、(b) 10 文目と 11 文目の間に「住む・場所」の枝を差し込み、前後の属性から当てはまる語を逆算する

着想: 2026-11-01 持ち主「最後の行の真っ白なおじいさんの属性から、人物を特定する流れを教えて」
      「10 と 11 行の間に、住む、場所、という属性の枝を作って、前後の属性から関連する文字列を逆算してくれる？」
(a) 人物の特定（実験87 の節グレインを使う）:
    1. 属性の語（真っ白・おじいさん）を枝葉に持つ節を探す（20 文目の「おじいさんに」の節）。
    2. その節の役割を見る: 述語「なる」に「に」で係る = 「X が Y に なる」の Y（何になったか）。Y は X のことを言っている。
    3. 同じ述語に係る主語の節（が・は・文頭の「太郎は、」）= X を取る → 太郎。
    4. 名前の名札で、ほかの文の同じ人物をまとめる（太郎 ⊂ 浦島太郎、1 文目）→ 浦島太郎。その人物の節グレインを全部集めた「人物の属性」も出す。
(b) 差し込む枝: 根 = 述語「住む」、枝 = 場所（に・で の役割、空き）。空きに入る語を逆算する。
    点 = 語板 v4 の「場所らしさ」（観点「に」「で」の 住む・暮らす の明るさ）× 前後の文の近さ（文章の中の名札、10.5 文目からの距離で薄く）
         + 前後の文の節とのつながり（同じ節の枝葉にいる語）。
    比べ: 文章を見ない（語板 v4 だけ）／ 文章だけ（近さだけ）。
結果: results/exp88.txt、docs/tree_identify_insert_88.md。実行: python experiments/exp88_tree_identify_insert.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.argv = [sys.argv[0]]
import importlib.util
spec = importlib.util.spec_from_file_location("e87", os.path.join(os.path.dirname(__file__), "exp87_story_tree_qa.py"))
src = open(spec.origin, encoding="utf-8").read().split("T = build(S)")[0]                  # 実験87 の部品（節グレインの作り方）だけを読み込む
ns = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), ns)
S, build, same, subtree_text, V4 = ns["S"], ns["build"], ns["same"], ns["subtree_text"], ns["V4"]
from goban_batch import run
sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"
T = build(S)
PROPER = {w[1] for s_ in S for b in s_["bunsetsu"] for w in b["words"] if "固有名詞" in w[2]}
L = ["# 文章樹: 人物の特定と、枝の差し込み（実験88、浦島太郎）", "", "作り方: experiments/exp88_tree_identify_insert.py（実験87 の節グレインを使う）。", ""]
# ---- (a) 人物の特定 ----
attrs = ["真っ白", "おじいさん"]
L += ["## (a) 「真っ白なおじいさん」から人物を特定する流れ", ""]
for nodes in T:
    for n in nodes:
        stext = subtree_text(nodes, n["id"])                                                  # 枝葉の文字列（おじいさん = 御 + 爺 + さん と分かれるので、文字列でも見る）
        if all(any(same(a, w) for w in n["sub"]) or a in stext for a in attrs) and n["head"] >= 0 and n["role"] != "どんな":
            par = nodes[n["head"]]
            L.append(f"1. 属性の語（{'・'.join(attrs)}）を枝葉に持つ節: {n['si'] + 1} 文目の「{n['text']}」（役割 = {n['role']}）")
            L.append(f"2. その節は 述語「{par['text']}」（{'・'.join(par['words'])}）に「{n['role']}」で係る。述語が「なる」で役割が「に」なので、"
                     f"「X が {n['text'].rstrip('に')} に なる」= X のことを言っている。")
            subj = [m for m in nodes if m["head"] == par["id"] and m["id"] != n["id"] and (m["role"] in ("が", "は") or m["dep"] in ("nsubj", "dislocated"))]
            subj.sort(key=lambda m: (0 if any(w for w in m["words"] if w in PROPER) else 1, m["id"]))   # 人の名前（固有名詞）の節を先に（「髪も」「髭も」も主語の形だが、人ではない）
            X = subj[0] if subj else None
            if X:
                name = [w for w in X["words"]][0] if X["words"] else X["text"]
                L.append(f"3. 同じ述語に係る主語の節: 「{X['text']}」（役割 = {X['role'] if X['role'] in ('が', 'は') else X['dep']}）→ X = **{name}**")
                names = set()                                                                # 続いた固有名詞をつないだ名前（浦島 + 太郎 → 浦島太郎）
                for s_ in S:
                    for b in s_["bunsetsu"]:
                        cur = ""
                        for w in b["words"] + [["", "", ""]]:
                            if "固有名詞" in w[2]:
                                cur += w[0]
                            else:
                                if cur:
                                    names.add(cur)
                                cur = ""
                alias = sorted(x for x in names if name in x and x != name)
                full = alias[0] if alias else name
                L.append(f"4. 名前の名札で、ほかの文の同じ人物をまとめる: {name} ⊂ {'・'.join(alias) or name} → **{full}**")
                prof = []
                for nn in T:
                    for m in nn:
                        if any(same(name, w) for w in m["words"]):
                            for c in m["children"]:
                                if nn[c]["role"] in ("どんな", "という"):
                                    prof.append(f"{nn[c]['text'].rstrip('、')}（{m['si'] + 1} 文目）")
                            if m["head"] >= 0:
                                prof.append(f"{m['role']} → {nn[m['head']]['text'].rstrip('。、')}（{m['si'] + 1} 文目）")
                L.append(f"5. 人物の属性（その人物の節が持つ枝・係る先、文の順）: " + "・".join(prof))
            break
# ---- (b) 10 文目と 11 文目の間に「住む・場所」の枝を差し込む ----
L += ["", "## (b) 10 文目と 11 文目の間に「住む・場所（空き）」の枝を差し込み、空きに入る語を逆算する", ""]
cands = {}                                                                                   # 文章に出た名詞と、その文の番号
for nodes in T:
    for n in nodes:
        for w in n["words"]:
            if w in V4:
                cands.setdefault(w, []).append(n["si"])
M = None
VW = {}
for k in (0, 1):
    M = torch.load(f"{PRIV}/models/goban_v4_part{k}_scratch.pt", weights_only=False)
    WI = {w: i for i, w in enumerate(M["words"])}
    with torch.no_grad():
        ws = [w for w in WI]
        place = []
        for i in range(0, len(ws), 1500):
            B = run(M["P"][[WI[w] for w in ws[i:i + 1500]]], 9, 499); B = B / B.sum(-1, keepdim=True)
            sc = torch.zeros(B.shape[0])
            for v, x in (("に", "住む"), ("で", "住む"), ("で", "暮らす"), ("に", "暮らす"), ("に", "帰る"), ("に", "戻る")):
                j = M["views"].index(v); r = M["pos"][v].get(x)
                if r is not None:
                    sc += B[:, j, r]
            place.append(sc)
        for w, s_ in zip(ws, torch.cat(place)):
            VW[w] = float(s_)
pos = 10.5                                                                                   # 差し込む位置（10 文目と 11 文目の間、文の番号は 1 から）
near = {w: sum(math.exp(-abs((si + 1) - pos) / 3) for si in sis) for w, sis in cands.items()}
ctxw = {w for si in (9, 10) for n in T[si] for w in n["sub"]}                                # 前後の文の節の枝葉にいる語
rows = []
for w in cands:
    pl = VW.get(w, 0.0)
    rows.append((w, pl, near[w], w in ctxw, pl * (0.2 + near[w]) + (0.3 * pl if w in ctxw else 0)))
top_all = sorted(VW.items(), key=lambda t: -t[1])[:12]
L.append("空きに入る語の候補（点 = 場所らしさ × 前後の近さ + 前後の文の節にいる語の上乗せ）:")
L.append("")
L += ["| 語 | 場所らしさ（語板 v4） | 前後の近さ | 前後の文の節にいる | 点 |", "|---|---|---|---|---|"]
for w, pl, nr, inc, sc in sorted(rows, key=lambda t: -t[4])[:12]:
    L.append(f"| {w} | {pl:.3f} | {nr:.2f} | {'○' if inc else ''} | {sc:.3f} |")
L += ["", "比べ: 文章を見ない（語板 v4 の場所らしさだけ、3 万語の上位）: " + "・".join(w for w, _ in top_all),
      "比べ: 文章だけ（前後の近さだけ）: " + "・".join(w for w, _ in sorted(near.items(), key=lambda t: -t[1])[:12])]
best = sorted(rows, key=lambda t: -t[4])[0][0]
L += ["", f"いちばん点の高い語を空きに入れた枝: 「{best}に 住む」（10 文目「故郷の母親のことが心配になり、帰ることを決意します。」の後）"]
open("docs/tree_identify_insert_88.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
with open("results/exp88.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(exp="88", lines=L), ensure_ascii=False) + "\n")
print("\n".join(L))
