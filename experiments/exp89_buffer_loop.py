"""実験89: バッファ語板を回して木を上り下りする — 融合・差分・光の重なりだけで質問に答えられるか（持ち主の案 2026-11-02）

着想: 持ち主「0. 開始時に 1 文章単位の語板をコピーして検索用のバッファ語板にする。1. 関係しそうな属性の比較 → バッファに反映。
      2. 構文による属性の変化（否定など、いまは無い）→ バッファに反映。3. 介入による真の関連属性の選定（差分だけ残す）→ バッファに反映。
      シンプルに 1〜3 を回すだけで木を上り下りして答えにたどり着けるか。真っ白なおじいさんから探すのと、文章全体から目指すのと、どちらが良いか」。
手札（実験87 の決まりの点付けは使わない）:
  属性の光 = 「この文章に出た語の名札」（1 文章単位の語板の名札、語ごとに 1 本）+「役割」（が・を・へ・という・どんな…、1 本ずつ）+「名詞の語板 v4 の光」を並べたもの。
  節の光 g(n) = 自分の語の名札 + 自分の役割 + 自分の名詞の語板の光。枝葉の光 G(n) = 自分と枝葉の g を足したもの（深いほど 0.7 倍、融合）。
  まわりの光 C(n) = G(親) − G(n)（差分: 係る先の枝葉のうち、自分の枝でない所 = 自分のまわり）+ 0.5 × G(文の根) − ...（文の根のまわり）。
  重なり = cos。
手順（質問 1 つ）:
  0. バッファ b = 質問の木の手がかりの節（疑問の語の節を除く）の光の融合（= 質問の 1 文章単位の語板のコピー）。
  2. 構文: 疑問の語から「答えの役割」の光 r を作る（何を → を、どこへ → へ・に、名前 → という …）。ここは手で決めた小さな表（学習器がまだ無い所）。
  A 上から（文章全体から目指す）:
     1. 各文の根の枝葉の光 G(根) と b の重なりで、文を選ぶ（幹へ移る）。
     3. 介入: その文の節 n ごとに、b のうち C(n) で説明できる分（重なり）× 役割の光 r との重なり。いちばん大きい節が答え。
        説明しきれない分（差分 b − 説明された分）が大きい（0.7 以上）なら、その文を外して 1 に戻る（最大 3 回）。
  B 下から（手がかりから上る）:
     1. b といちばん重なる節（手がかりの語そのものの節）を探す → 係る先（親）へ上る。
     3. 介入: 親に係る節（兄弟）のうち、役割の光 r と重なり、かつ b との差分を減らす節を答えにする。減らなければ、名前の名札が同じ節（ほかの文）へ移って繰り返す（最大 3 回）。
質問: 実験87 の 20 問 + 直すのに使っていない 10 問。
結果: results/exp89.txt、docs/buffer_loop_89.md。実行: python experiments/exp89_buffer_loop.py
"""
import json, math, os, sys

import torch

spec_path = os.path.join(os.path.dirname(__file__), "exp87_story_tree_qa.py")
src = open(spec_path, encoding="utf-8").read().split("T = build(S)")[0]                      # 実験87 の部品（木・節・語板 v4 の光・答えの文字列）を借りる
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
S, build, subtree_text, V4, WH, CONTENT = ns["S"], ns["build"], ns["subtree_text"], ns["V4"], ns["WH"], ns["CONTENT"]
sys.stdout.reconfigure(encoding="utf-8")
T = build(S)
Q1 = json.load(open("data/urashima_qa_parse.json", encoding="utf-8")); A1 = [l.rstrip("\n").split("\t")[1].split("|") for l in open("data/urashima_qa.tsv", encoding="utf-8")]
Q2 = json.load(open("data/urashima_qa_heldout_parse.json", encoding="utf-8")); A2 = [l.rstrip("\n").split("\t")[1].split("|") for l in open("data/urashima_qa_heldout.tsv", encoding="utf-8")]
# ---- 属性の光の並び: 名札（文章と質問に出た語）+ 役割 + 語板 v4 ----
VOC = sorted({w for nodes in T for n in nodes for w in n["words"]} | {w[1] for q in Q1 + Q2 for b in q["bunsetsu"] for w in b["words"] if w[2].startswith(CONTENT)})
VI = {w: i for i, w in enumerate(VOC)}
ROLES = sorted({n["role"] for nodes in T for n in nodes} | {r for rs in WH.values() for r in rs}); RI = {r: i for i, r in enumerate(ROLES)}
D4 = next(iter(V4.values())).shape[0]; DIM = len(VOC) + len(ROLES) + D4


def tag_of(w):
    """名札: 同じ語に加え、名前の一部（太郎 ⊂ 浦島太郎）も同じ名札に少し光らせる（文をまたぐ人物のまとめ）。"""
    v = torch.zeros(DIM)
    for u, i in VI.items():
        if u == w:
            v[i] = 1.0
        elif len(w) >= 2 and len(u) >= 1 and (w in u or u in w):
            v[i] = 0.7
    return v


def own(n_words, role=None, w_light=0.5):
    v = torch.zeros(DIM)
    for w in n_words:
        v += tag_of(w)
        if w in V4:
            v[len(VOC) + len(ROLES):] += w_light * V4[w]
    if role is not None and role in RI:
        v[len(VOC) + RI[role]] += 1.0
    return v


for nodes in T:                                                                              # 節の光 g・枝葉の光 G・まわりの光 C
    for n in nodes:
        n["g"] = own(n["words"], n["role"])
    for n in nodes:
        G = torch.zeros(DIM); stack = [(n["id"], 1.0)]
        while stack:
            i, w = stack.pop(); G += w * nodes[i]["g"]; stack += [(c, w * 0.7) for c in nodes[i]["children"]]
        n["G"] = G
    root = next(m for m in nodes if m["head"] == -1)
    for n in nodes:
        par = nodes[n["head"]] if n["head"] >= 0 else None
        n["C"] = (par["G"] - n["G"] if par else torch.zeros(DIM)) + 0.5 * (root["G"] - n["G"] if n is not root else 0)   # 差分でまわりを作る
cos = lambda a, b: float(a @ b / (a.norm() * b.norm()).clamp(min=1e-9))
ROLE_PART = slice(len(VOC), len(VOC) + len(ROLES))


def question(q):
    txt = q["sentence"]
    wh = next((k for k in ("名前", "何を", "何に", "何が", "何と", "どこへ", "どこで", "誰", "どんな", "何") if k in txt), "何")
    r = torch.zeros(DIM)
    for x in WH[wh]:
        if x in RI:
            r[len(VOC) + RI[x]] = 1.0                                                         # 2. 構文 → 答えの役割の光（手で決めた表）
    stop = {"何", "誰", "どこ", "名前", "どんな", "最後"}
    b = torch.zeros(DIM)
    for bb in q["bunsetsu"]:
        ws = [w[1] for w in bb["words"] if w[2].startswith(CONTENT) and w[1] not in stop and w[0] not in stop]
        b += own(ws, None)                                                                    # 0. バッファ = 質問の手がかりの光の融合（役割は入れない）
    return b, r


def top_down(b, r):
    """A 上から: 文を選び（幹へ移る）、その文の中で介入（差分）して答えの節を決める。"""
    banned = set(); trace = []
    for it in range(3):
        si = max((i for i in range(len(T)) if i not in banned), key=lambda i: cos(next(m for m in T[i] if m["head"] == -1)["G"], b))
        nodes = T[si]; best = (-1, None)
        for n in nodes:
            rr = float(n["g"][ROLE_PART] @ r[ROLE_PART])
            if rr <= 0:
                continue
            expl = cos(n["C"], b)                                                             # b のうち、この節のまわりで説明できる分
            if rr * expl > best[0]:
                best = (rr * expl, n)
        trace.append(f"{si + 1} 文目")
        if best[1] is not None and best[0] >= 0.3:                                            # 差分（説明しきれない分 = 1 − 重なり）が 0.7 以下なら、ここで答える
            return best[1], nodes, trace
        banned.add(si)
    return (best[1], nodes, trace) if best[1] is not None else (None, None, trace)


def bottom_up(b, r):
    """B 下から: 手がかりの節から係る先へ上り、兄弟のうち役割が合い、差分を減らす節を答えにする。届かなければ名札の同じ別の文の節へ移る。"""
    cands = sorted(((cos(n["g"], b), si, n["id"]) for si, nodes in enumerate(T) for n in nodes), reverse=True)
    trace = []
    for sc, si, ni in cands[:3]:
        nodes = T[si]; n = nodes[ni]; trace.append(f"{si + 1} 文目「{n['text']}」")
        par = nodes[n["head"]] if n["head"] >= 0 else n                                        # 1. 係る先（幹）へ上る
        resid0 = 1 - cos(par["G"], b)
        best = (-1, None)
        for m in nodes:
            if m["head"] != par["id"] or m["id"] == ni:
                continue
            rr = float(m["g"][ROLE_PART] @ r[ROLE_PART])
            if rr <= 0:
                continue
            gain = cos(m["C"], b)                                                             # 3. 介入: この兄弟を答えにしたとき、まわりが b を説明する分
            if rr * gain > best[0]:
                best = (rr * gain, m)
        if best[1] is not None:
            return best[1], nodes, trace
    return None, None, trace


def run_all(Q, A):
    out = []
    for q, ans in zip(Q, A):
        b, r = question(q)
        row = dict(q=q["sentence"], gold="・".join(ans))
        for name, fn in (("上から", top_down), ("下から", bottom_up)):
            n, nodes, tr = fn(b, r)
            a = subtree_text(nodes, n["id"]) if n else "（なし）"
            row[name] = a; row[name + "_ok"] = any(x in a for x in ans) or (a != "（なし）" and any(a in x for x in ans)); row[name + "_道"] = " → ".join(tr)
        out.append(row)
    return out


R1 = run_all(Q1, A1); R2 = run_all(Q2, A2)
res = dict(exp="89", 上から_20=sum(r["上から_ok"] for r in R1), 下から_20=sum(r["下から_ok"] for r in R1),
           上から_10=sum(r["上から_ok"] for r in R2), 下から_10=sum(r["下から_ok"] for r in R2), rows=R1 + R2)
with open("results/exp89.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
L = ["# バッファ語板を回して木を上り下りする（実験89、浦島太郎）", "", "作り方: experiments/exp89_buffer_loop.py。実験87 の決まりの点付けを使わず、融合（足す）・差分（引く）・光の重なり（cos）だけで答える。",
     "手で決めた所: 文の木（係り受け）と、疑問の語 → 答えの役割（2. 構文の所、学習器がまだ無い）。", "",
     f"| | 質問 20（実験87 の決まりでは 19） | 使っていない質問 10（決まりでは 7） |", "|---|---|---|",
     f"| A 上から（文章全体から目指す） | {res['上から_20']} | {res['上から_10']} |", f"| B 下から（手がかりから上る） | {res['下から_20']} | {res['下から_10']} |", "",
     "| 質問 | 正解 | 上から | 道 | 下から | 道 |", "|---|---|---|---|---|---|"]
for r in R1 + R2:
    L.append(f"| {r['q']} | {r['gold']} | {r['上から']} {'○' if r['上から_ok'] else '×'} | {r['上から_道']} | {r['下から']} {'○' if r['下から_ok'] else '×'} | {r['下から_道']} |")
open("docs/buffer_loop_89.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L[:9]))
