"""実験87: 文章樹と節グレイン — 浦島太郎の 20 文を木にし、節ごとにその下の枝葉の属性を持たせ、質問で木をたどれるか・介入で答えが変わるかを試す

着想: 2026-11-01 持ち主「節の部分ごとに節グレインを作ってほしい（属性樹と言ってもいい）。節にはその下の枝葉の属性を持つ。これを全文に。
      最後に質問で文章をたぐれるかを試験し、理解しているかを介入処理で確認したい。『漁師の名前は？』なら『浦島太郎という』『浦島』『太郎』『浦島太郎』が正解」。
作り方:
  木: GiNZA（係り受け）で文を文節に分け、文節がどの文節に係るかで木にする（根 = 述語。data/urashima_parse.json、非公開フォルダの tools/parse_bunsetsu.py）。
  節グレイン: 文節ごとに「自分の語（正規形）・役割（が・は・を・に・へ・で・と・の・という・連体修飾・引用）」と、
      その下の枝葉の語をまとめた属性（深い枝ほど薄く）と、名詞の語板 v4 の光を明るさで足した「節の光」を持つ。
  人物などのまとめ（文をまたぐ）: 同じ語や、名前の一部（太郎 ⊂ 浦島太郎、箱 ⊂ 玉手箱）は同じものとして扱う（短期の記憶の名札）。
  質問: 質問も同じように木にし、疑問の語（何・誰・どこ・どんな・名前・何と）から「答えになる節の役割」を決め、質問の残りの語（手がかり）が
      同じ文の木の中にあるか（同じ語・名前の一部・語板 v4 の光の重なり）で点を付け、いちばん点の高い節の枝葉をまとめて答えにする。
  正解: 答えに正解の語のどれかが入っていれば ○。
  介入: 節グレインの中身を書き換えて（浦島太郎 → 桃太郎、玉手箱 → 宝石箱、歓迎した 乙姫様 → 亀）、同じ質問の答えがそれに合わせて変わるか、
      関係のない質問の答えは変わらないかを確かめる。
限界: 疑問の語から役割への対応（「何を」→ を、「どこへ」→ へ・に）は手で決めた小さな表。受け身（いじめられている亀 → いじめたのは に の節）も手で 1 つ足した。
結果: results/exp87.txt、docs/story_tree_87.md。実行: python experiments/exp87_story_tree_qa.py
"""
import copy, json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"
S = json.load(open("data/urashima_parse.json", encoding="utf-8"))
Q = json.load(open("data/urashima_qa_parse.json", encoding="utf-8"))
ANS = [l.rstrip("\n").split("\t")[1].split("|") for l in open("data/urashima_qa.tsv", encoding="utf-8")]
CONTENT = ("名詞", "動詞", "形容詞", "形状詞", "代名詞")
# ---- 語板 v4（名詞の光）: 物語に出る名詞だけ光を通す ----
V4 = {}
for k in (0, 1):
    M = torch.load(f"{PRIV}/models/goban_v4_part{k}_scratch.pt", weights_only=False)
    WI = {w: i for i, w in enumerate(M["words"])}
    need = {w[1] for s in S + Q for b in s["bunsetsu"] for w in b["words"] if w[2].startswith("名詞") and w[1] in WI}
    if need:
        with torch.no_grad():
            Bv = run(M["P"][[WI[w] for w in need]], 9, 499)
        Bv = Bv / Bv.sum(-1, keepdim=True)
        for w, x in zip(need, Bv[:, :, :490].reshape(len(need), -1)):
            V4[w] = torch.nn.functional.normalize(x, dim=0)
light_sim = lambda a, b: float(V4[a] @ V4[b]) if a in V4 and b in V4 else 0.0


def role(b):
    """文節の役割: 最後の助詞（という は 1 つ）、または係り方（連体修飾 acl/amod・引用 = と で終わる述語）。"""
    ws = b["words"]; tags = [w[2] for w in ws]; txt = b["text"]
    if "という" in txt or "と言う" in txt:
        return "という"
    parts = [w[0] for w in ws if w[2].startswith("助詞") and w[0] not in ("て", "は") ]
    if b["dep"] in ("acl", "amod") or (ws and ws[-1][2].startswith(("形容詞", "形状詞")) and b["dep"] != "ROOT"):
        return "どんな"
    if txt.rstrip("、，。 」").endswith("と") and any(t.startswith(("動詞", "助動詞")) for t in tags):
        return "引用"
    ps = [w[0] for w in ws if w[2].startswith("助詞")]
    for p_ in ps:                                                                            # 格の助詞を先に見る（「城では」= で、「浜辺へと」= へ）
        if p_ in ("が", "を", "に", "へ", "で", "から"):
            return p_
    for p_ in reversed(ps):
        if p_ in ("は", "と", "の", "も"):
            return p_
    return b["dep"]


def build(S):
    """節グレイン: 文ごと・文節ごとに 自分の語・役割・子・枝葉の語（深さで薄く）・節の光。"""
    T = []
    for si, s in enumerate(S):
        B = s["bunsetsu"]; nodes = []
        for b in B:
            nodes.append(dict(si=si, id=b["id"], text=b["text"], head=b["head"], role=role(b), dep=b["dep"],
                              words=[w[1] for w in b["words"] if w[2].startswith(CONTENT)], passive=any(w[1] in ("れる", "られる") for w in b["words"]),
                              children=[c["id"] for c in B if c["head"] == b["id"]]))
        for n in nodes:                                                                       # その下の枝葉の属性（深い枝ほど薄く）と、節の光
            sub = {}; stack = [(n["id"], 1.0)]
            while stack:
                i, wgt = stack.pop()
                for w in nodes[i]["words"]:
                    sub[w] = max(sub.get(w, 0), wgt)
                stack += [(c, wgt * 0.7) for c in nodes[i]["children"]]
            n["sub"] = sub
            ls = [V4[w] * g for w, g in sub.items() if w in V4]
            n["light"] = torch.nn.functional.normalize(sum(ls), dim=0) if ls else None
        T.append(nodes)
    return T


def same(a, b):
    """同じものか: 同じ語、名前の一部（太郎 ⊂ 浦島太郎、箱 ⊂ 玉手箱）、語板の光がとても近い。"""
    return a == b or (len(a) >= 2 and len(b) >= 1 and (a in b or b in a)) or light_sim(a, b) > 0.9


def subtree_text(nodes, i):
    ids = []; stack = [i]
    while stack:
        j = stack.pop(); ids.append(j); stack += nodes[j]["children"]
    t = "".join(nodes[j]["text"] for j in sorted(ids))
    for suf in ("、", "。", "，", "が", "は", "を", "に", "へ", "で", "と", "の", "も", "へと", "には"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    return t.strip("「」、")


WH = {"何を": {"を", "も"}, "何に": {"に"}, "何が": {"が", "は"}, "何と": {"引用"}, "どこへ": {"へ", "に"}, "どこで": {"で"}, "誰": {"が", "は", "に"},
      "どんな": {"どんな"}, "名前": {"という"}, "何": {"を", "が", "に", "は"}}


def ask(T, q):
    """質問の木 q から、答えの節を探す。"""
    txt = q["sentence"]; qn = [w for b in q["bunsetsu"] for w in b["words"]]
    wh = next((k for k in ("名前", "何を", "何に", "何が", "何と", "どこへ", "どこで", "誰", "どんな", "何") if k in txt), "何")
    roles = WH[wh]
    stop = {"何", "誰", "どこ", "名前", "どんな", "最後"}
    anchors = [w[1] for w in qn if w[2].startswith(CONTENT) and w[1] not in stop and w[0] not in stop]
    qverb = [w[1] for w in qn if w[2].startswith("動詞") and w[1] not in ("為る", "居る", "成る", "有る")]
    if wh == "名前":                                                                          # 「X の名前」: X に係る「という」の節
        roles = {"という"}
    best = (-1, None)
    for nodes in T:
        for n in nodes:
            r = n["role"]
            ok = r in roles
            if wh == "誰" and r == "に":                                                     # 受け身: 「〜られている X」の いじめた人 = に の節
                ok = n["head"] >= 0 and nodes[n["head"]]["passive"]
            if wh == "どんな":
                ok = r == "どんな"
            if not ok:
                continue
            score = 0.0
            ctx = {}                                                                         # 答えの候補の まわり（親とその枝葉、ただし自分の枝は除く）
            par = nodes[n["head"]] if n["head"] >= 0 else None
            root = next(m for m in nodes if m["head"] == -1)
            for m, mw in (([(par, 1.0), (root, 0.5)]) if par else [(root, 1.0)]):            # 係る先（親）のまわりを強く、文の根のまわりは半分
                for w, g in m["sub"].items():
                    if w not in n["sub"]:
                        ctx[w] = max(ctx.get(w, 0), g * mw)
            for a in anchors:
                hit = max([g for w, g in ctx.items() if same(a, w)] + [0])
                score += 1.2 * hit
            for v in qverb:                                                                  # 述語が合えば強く
                if par and any(same(v, w) for w in par["words"]):
                    score += 1.0
                elif any(same(v, w) for w in root["words"]):
                    score += 1.0
            if r in ("が", "を", "へ", "に", "で"):                                           # 格の助詞がはっきりしている節を少し優先（は・も より）
                score += 0.3
            if wh == "どんな" and par:                                                      # 「どんな X」: 係る先が X
                score += 2.0 * max([1.0 if same(a, w) else 0 for a in anchors for w in par["words"]] + [0])
            if score > best[0]:
                best = (score, (n, nodes))
    if best[1] is None:
        return "（見つからない）", None
    n, nodes = best[1]
    return subtree_text(nodes, n["id"]), n


def evaluate(T, tag):
    rows = []
    for q, ans in zip(Q, ANS):
        a, n = ask(T, q)
        ok = any(x in a for x in ans) or any(a and a in x for x in ans)
        rows.append(dict(q=q["sentence"], ans=a, gold="・".join(ans), ok=ok, sent=(n["si"] + 1) if n else None))
    return rows


T = build(S)
base = evaluate(T, "元")
acc = sum(r["ok"] for r in base) / len(base)
print("正解", sum(r["ok"] for r in base), "/", len(base))


def intervene(fn):
    S2 = copy.deepcopy(S); fn(S2); return evaluate(build(S2), "介入")


def rename(S2, si, old, new):
    for b in S2[si]["bunsetsu"]:
        if old in b["text"]:
            b["text"] = b["text"].replace(old, new); b["words"] = [[w[0].replace(old, new), w[1].replace(old, new), w[2]] for w in b["words"]]
            if old in "".join(w[1] for w in b["words"]) or True:
                pass


def merge_name(S2, si, old_parts, new):
    """「浦島」「太郎」の 2 語を 1 つの新しい名前に書き換える（節グレインの中身の書き換え）。"""
    for b in S2[si]["bunsetsu"]:
        if all(p in b["text"] for p in old_parts):
            b["text"] = b["text"].replace("".join(old_parts), new)
            b["words"] = [[new, new, "名詞-固有名詞-人名"]] + [w for w in b["words"] if w[0] not in old_parts]


INT = [("1 文目の名前を 浦島太郎 → 桃太郎", lambda S2: merge_name(S2, 0, ["浦島", "太郎"], "桃太郎"), [0], [2, 7]),
       ("12 文目の 玉手箱 → 宝石箱", lambda S2: rename(S2, 11, "玉手箱", "宝石箱"), [12], [16]),
       ("7 文目の 乙姫様 → 亀（歓迎したのは）", lambda S2: rename(S2, 6, "乙姫様", "亀"), [8], [11])]
irows = []
for name, fn, qs, others in INT:
    R = intervene(fn)
    for qi in qs:
        irows.append(dict(介入=name, 質問=Q[qi]["sentence"], 前=base[qi]["ans"], 後=R[qi]["ans"], 変わった=base[qi]["ans"] != R[qi]["ans"], 種類="変わるはず"))
    for qi in others:
        irows.append(dict(介入=name, 質問=Q[qi]["sentence"], 前=base[qi]["ans"], 後=R[qi]["ans"], 変わった=base[qi]["ans"] != R[qi]["ans"], 種類="変わらないはず"))
# 決まりを直すのに使っていない質問 10（直したあとに作って、1 回だけ測る）
Q_ALL, ANS_ALL = Q, ANS
Q = json.load(open("data/urashima_qa_heldout_parse.json", encoding="utf-8"))
ANS = [l.rstrip("\n").split("\t")[1].split("|") for l in open("data/urashima_qa_heldout.tsv", encoding="utf-8")]
held = evaluate(T, "未使用")
Q, ANS = Q_ALL, ANS_ALL
res = dict(exp="87", questions=len(base), correct=sum(r["ok"] for r in base), rows=base, interventions=irows,
           heldout_correct=sum(r["ok"] for r in held), heldout=held)
with open("results/exp87.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
L = ["# 文章樹と節グレイン（実験87、浦島太郎 20 文）", "", "持ち主の依頼（2026-11-01）。作り方: experiments/exp87_story_tree_qa.py。",
     f"質問 20: **{res['correct']} / 20 正解**。", "", "| 質問 | 答え（木から） | 正解 | 文 | |", "|---|---|---|---|---|"]
for r in base:
    L.append(f"| {r['q']} | {r['ans']} | {r['gold']} | {r['sent']} | {'○' if r['ok'] else '×'} |")
L += ["", f"## 決まりを直すのに使っていない質問（10 問、1 回だけ測った）: **{res['heldout_correct']} / 10**", "", "| 質問 | 答え（木から） | 正解 | 文 | |", "|---|---|---|---|---|"]
for r in held:
    L.append(f"| {r['q']} | {r['ans']} | {r['gold']} | {r['sent']} | {'○' if r['ok'] else '×'} |")
L += ["", "## 介入（節グレインの中身を書き換えて、同じ質問をする）", "", "| 介入 | 質問 | 前 | 後 | 期待 | |", "|---|---|---|---|---|---|"]
for r in irows:
    good = r["変わった"] if r["種類"] == "変わるはず" else not r["変わった"]
    L.append(f"| {r['介入']} | {r['質問']} | {r['前']} | {r['後']} | {r['種類']} | {'○' if good else '×'} |")
open("docs/story_tree_87.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
