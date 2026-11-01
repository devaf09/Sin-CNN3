"""実験95: 学んだフィルターを木の答え探しに組み込む — 手で書いた決まりをどこまで減らせるか（浦島太郎の質問 30）

着想: 2026-11-02 持ち主「思考のデジタル重みが大きいから、文法に沿った学習器を追加して少しずつ減らしたい」。
組み合わせ（どれも実験89 の「上から」= 文の幹を選び、まわりの光（差分）× 役割の合い で節を選ぶ）:
  手: 文節の役割 = 手の決まり（実験87 の role）、疑問 → 役割 = 手の表。
  疑問フィルター: 疑問 → 役割 だけ 実験90 の光のフィルター。
  両方: 疑問フィルター + 文節の役割も 実験92 の役割フィルター（係り方 nsubj・obj・obl・acl・nmod・advcl…）。
       役割フィルターの係り方と、疑問フィルターの答えの役割（が・を・へ・どんな・という・引用…）の対応は小さな表（どの係り方がどの役割に当たるか）。
       文節の中の助詞（へ・で・から）は、文節の語そのもの（決まりではなく材料）として使う。
残る手の所: 文の木（GiNZA）、役割の対応の小さな表、名前の名札（太郎 ⊂ 浦島太郎）、答えの文字列のまとめ方。
結果: results/exp95.txt。実行: python experiments/exp95_filters_in_tree.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import _h, VAL0

spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
T, S, cos, subtree_text, question, Q1, A1, Q2, A2 = ns["T"], ns["S"], ns["cos"], ns["subtree_text"], ns["question"], ns["Q1"], ns["A1"], ns["Q2"], ns["A2"]
sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"
role_hand = ns["build"].__globals__["role"]


def light_filter(path):
    """保存した光のフィルター（縦 512 × 板 16、参照の光あり）を読み、特徴の並び → 出力の割合 を返す関数にする。"""
    M = torch.load(path, weights_only=False); P = M["P"]; FEATS = M["feats"]; LABS = M["labels"]; NF = len(FEATS); N, K = P.shape
    H = _h("cpu", N); yy = torch.arange(N, dtype=torch.float32)
    WAVE = torch.stack([torch.exp(1j * 2 * math.pi * (k + 1) * yy / N) / math.sqrt(N) for k in range(NF + 1)]).to(torch.complex64)
    FI = {f: i for i, f in enumerate(FEATS)}

    def f(feat_names):
        x = torch.zeros(1, NF)
        for nm in feat_names:
            if nm in FI:
                x[0, FI[nm]] = 1.0
        u = x.to(torch.complex64) @ WAVE[:NF]; u = u / u.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-9)
        u = torch.nn.functional.pad((u + WAVE[NF][None]) / math.sqrt(2), (0, N))
        with torch.no_grad():
            for k in range(K):
                u = torch.cat([u[..., :N] * torch.exp(1j * P[:, k]), u[..., N:]], -1)
                u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
            o = torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs()[0, VAL0:VAL0 + len(LABS)] ** 2
        return dict(zip(LABS, (o / o.sum()).tolist()))
    return f


ROLE_F = light_filter(f"{PRIV}/models/role_filter.pt"); WH_F = light_filter(f"{PRIV}/models/wh_role_filter.pt")
fn = lambda w: w[2].startswith(("助詞", "助動詞")); pos2 = lambda t: "-".join(t.split("-")[:2])


def role_feats(b, head):
    ws = b["words"]; f = ["fw:" + w[0] for w in ws if fn(w)]
    cont = [w for w in ws if not fn(w) and not w[2].startswith("補助記号")] or ws
    f += ["first:" + pos2(cont[0][2]), "last:" + pos2(ws[-1][2])]
    if head is None:
        f.append("root")
    else:
        hc = [w for w in head["words"] if not fn(w) and not w[2].startswith("補助記号")] or head["words"]
        f += ["hfirst:" + pos2(hc[0][2])] + ["hfw:" + w[0] for w in head["words"] if fn(w)]
    return f


for si, nodes in enumerate(T):                                                               # 文節ごとに、役割フィルターの係り方の割合
    B = S[si]["bunsetsu"]
    for n in nodes:
        b = B[n["id"]]; head = B[b["head"]] if b["head"] >= 0 else None
        n["dep_p"] = ROLE_F(role_feats(b, head))
        n["parts"] = [w[0] for w in b["words"] if w[2].startswith("助詞")]
# 答えの役割（疑問フィルターの出力）と、役割フィルターの係り方の対応（小さな表）
ALIGN = {"が": [("nsubj", None)], "は": [("nsubj", None)], "も": [("nsubj", None), ("obl", None)], "を": [("obj", None)], "に": [("obl", "に")], "へ": [("obl", "へ")],
         "で": [("obl", "で")], "から": [("obl", "から")], "と": [("obl", "と")], "どんな": [("acl", None), ("amod", None)], "の": [("nmod", "の")],
         "という": [("nmod", "と")], "引用": [("advcl", "と"), ("ccomp", "と")], "advmod": [("advmod", None)], "obl": [("obl", None)]}


def compat_filter(n, target):
    """節が答えの役割に合う度合い = Σ 答えの役割の割合 × 係り方の割合（助詞の指定があれば、その助詞が文節にあるときだけ）。"""
    s = 0.0
    for lab, p in target.items():
        for dep, part in ALIGN.get(lab, []):
            if part is None or part in n["parts"]:
                s += p * n["dep_p"].get(dep, 0.0)
    return s


q_feats_src = open(os.path.join(os.path.dirname(__file__), "exp90_wh_role_filter.py"), encoding="utf-8").read()
i0 = q_feats_src.index("def q_feats(q):"); i1 = q_feats_src.index("ROLES89, RI89, VOC")
ns90 = {"role89": role_hand}; exec(q_feats_src[i0:i1], ns90); q_feats = ns90["q_feats"]


def wh_target(q):
    d = q_feats(q)
    f = ["wh:" + d["wh"], "part:" + d["part"]] + (["cleft"] if d["cleft"] else []) + (["passive"] if d["passive"] else [])
    f += ["other:" + r for r in d["others"]] + ["verb:" + d["verb"]]
    t = WH_F(f)
    if d["part"] == "という":
        t = {"という": 1.0}
    return t


def answer(q, mode):
    b, r_hand = question(q)
    if mode == "手":
        n, nodes, _ = ns["top_down"](b, r_hand)
        return subtree_text(nodes, n["id"]) if n else "（なし）"
    target = wh_target(q)
    banned = set()
    for it in range(3):
        si = max((i for i in range(len(T)) if i not in banned), key=lambda i: cos(next(m for m in T[i] if m["head"] == -1)["G"], b))
        best = (-1, None)
        for n in T[si]:
            if mode == "疑問フィルター":
                rr = sum(p for lab, p in target.items() if lab == n["role"])                    # 文節の役割は手の決まりのまま
            else:
                rr = compat_filter(n, target)                                                    # 文節の役割も役割フィルター
            sc = rr * cos(n["C"], b)
            if sc > best[0]:
                best = (sc, n)
        if best[1] is not None and best[0] >= 0.1:
            return subtree_text(T[si], best[1]["id"])
        banned.add(si)
    return subtree_text(T[si], best[1]["id"]) if best[1] is not None else "（なし）"


res = dict(exp="95")
for mode in ("手", "疑問フィルター", "両方"):
    ok20 = ok10 = 0; rows = []
    for k, (Q, A) in enumerate(((Q1, A1), (Q2, A2))):
        for q, ans in zip(Q, A):
            a = answer(q, mode); ok = any(x in a for x in ans) or (a != "（なし）" and any(a in x for x in ans))
            rows.append(dict(q=q["sentence"], ans=a, ok=ok))
            if k == 0:
                ok20 += ok
            else:
                ok10 += ok
    res[mode] = dict(q20=ok20, q10=ok10, rows=rows)
    print(mode, "20 問", ok20, "未使用 10 問", ok10, flush=True)
with open("results/exp95.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
