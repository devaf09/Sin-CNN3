"""実験96: 文章樹の根・幹・枝に重みを付ける（持ち主の問い 2026-11-02「根と枝で属性重みを変えてる？」→「うん（試して）」）

いまの木（実験87・89）: 節の光は根でも枝でも同じ重み。枝葉の光は深さで 0.7 倍ずつ薄くするだけ。
試す重み（節の光 g に掛けてから、枝葉の光 G・まわりの光 C を作り直す）:
  役割の重み: 述語（根）1.5、主語（が・は）1.3、目的語（を）1.2、場所・相手（に・へ・で・から・と）1.0、名前（という）1.0、修飾（どんな・の）0.7、そのほか 0.8。
  結びつきの重み: 材料（実験84 と同じ会話・青空文庫・Web 小説）で、その語のあと 3 語以内に動詞が来る割合（動詞との結びつき）を数え、
      節の重み = 0.5 + 節の語の結びつきのいちばん大きい値（述語は 1.5）。実験84 の「重要度のグラデーション」と同じ考え。
  両方: 役割の重み × 結びつきの重み。
比べ: 重みなし（実験89 の「上から」）。浦島太郎の質問 20 + 使っていない 10。
結果: results/exp96.txt。実行: python experiments/exp96_tree_weights.py
"""
import collections, json, os, pickle, sys

import torch

spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
T, DIM, run_all, Q1, A1, Q2, A2 = ns["T"], ns["DIM"], ns["run_all"], ns["Q1"], ns["A1"], ns["Q2"], ns["A2"]
sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"
for nodes in T:
    for n in nodes:
        n["g0"] = n["g"].clone()
# 動詞との結びつき（材料で数える）
DOCS = pickle.load(open(f"{PRIV}/data/views/synth_units.pkl", "rb"))
near = collections.Counter(); seen = collections.Counter()
for s, rows in DOCS:
    if s != "train":
        continue
    for r in rows:
        for k, (w, l) in enumerate(r):
            seen[w] += 1
            if any(r[j][1].startswith("動詞") for j in range(k + 1, min(len(r), k + 4))):
                near[w] += 1
del DOCS
link = lambda w: near[w] / seen[w] if seen[w] >= 5 else 0.3
ROLE_W = {"ROOT": 1.5, "が": 1.3, "は": 1.3, "を": 1.2, "に": 1.0, "へ": 1.0, "で": 1.0, "から": 1.0, "と": 1.0, "という": 1.0, "どんな": 0.7, "の": 0.7}


def rebuild(weight_fn):
    """節の光に重みを掛け、枝葉の光 G・まわりの光 C を作り直す（実験89 と同じ作り方）。"""
    for nodes in T:
        for n in nodes:
            n["g"] = n["g0"] * weight_fn(n)
        for n in nodes:
            G = torch.zeros(DIM); stack = [(n["id"], 1.0)]
            while stack:
                i, w = stack.pop(); G += w * nodes[i]["g"]; stack += [(c, w * 0.7) for c in nodes[i]["children"]]
            n["G"] = G
        root = next(m for m in nodes if m["head"] == -1)
        for n in nodes:
            par = nodes[n["head"]] if n["head"] >= 0 else None
            n["C"] = (par["G"] - n["G"] if par else torch.zeros(DIM)) + 0.5 * (root["G"] - n["G"] if n is not root else 0)


role_w = lambda n: 1.5 if n["head"] == -1 else ROLE_W.get(n["role"], 0.8)
link_w = lambda n: 1.5 if n["head"] == -1 else 0.5 + max([link(w) for w in n["words"]] + [0.0])
MODES = {"重みなし": lambda n: 1.0, "役割の重み": role_w, "結びつきの重み": link_w, "両方": lambda n: role_w(n) * link_w(n)}
res = dict(exp="96")
for name, fn in MODES.items():
    rebuild(fn)
    R1 = run_all(Q1, A1); R2 = run_all(Q2, A2)
    res[name] = dict(q20=sum(r["上から_ok"] for r in R1), q10=sum(r["上から_ok"] for r in R2), wrong=[r["q"] for r in R1 + R2 if not r["上から_ok"]])
    print(name, "20 問", res[name]["q20"], "未使用 10 問", res[name]["q10"], "外れ", res[name]["wrong"], flush=True)
with open("results/exp96.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
