"""実験63: 相関から因果へ — 「緑のトマトは酸っぱい」の原因は色か、熟し具合か。「赤いトマトを緑に塗ったら酸っぱくなるか？」に答える

着想: 2026-10-29 持ち主のアイディア ③。赤いリンゴ・青いリンゴ・赤いトマト・緑のトマトの属性から、
      「緑のトマトは赤いトマトより酸っぱい」を相関し、「緑のリンゴは赤いリンゴより酸っぱい」は成り立たない（不確実）ことから、
      「緑 → 酸っぱい」ではなく「未熟 → 酸っぱい」と判断したい。質問「赤いトマトを緑色に塗ったら酸っぱくなるか？」に
      「色は関係ないと思います、未熟でないなら甘酸っぱいかもしれません」と答えたい。
      統計学の「交絡」（未熟が、緑色と酸っぱさの両方の原因）と「介入」（塗る = 色だけを変える。熟し具合は変わらない）の問題。
方法: 物の状態ごとに 1 枚のグレイン（src/grain.py、観点 = 色・味・熟し具合・種類）。表（下の TABLE）を学習。3 seed。
  1. 差分（学び 44: 明るさの引き算）: 同じ種類で状態だけ違う組（赤いトマトと緑のトマト など）の、観点ごとの明るさの差 Δ_観点 = |明るさの差| の和の半分。
  2. 原因の見つけ方（ミルの一致法・差異法を光の明るさで）: 味の差 Δ味 が、どの観点の差と一緒に起きるかを、組全体で比べる。
       原因らしさ(観点 c) = Σ_組 Δc·Δ味 / √(Σ Δc² · Σ Δ味²)（一緒に変わるほど 1。c が変わっても味が変わらない組があると下がる）
     比べ: 組を 1 つ（トマトだけ）しか見ないと、色と熟し具合の区別がつかない（どちらも 1）。
  3. 不確実性（持ち主の ①）: 色の差に対する味の差が、組ごとにばらつく（トマトでは大、リンゴでは 0）→「色 → 酸っぱい」は不確実。
  4b. 打ち消し合いを考える形: Δ味 ≈ a·Δ色 + b·Δ熟し具合 を、すべての組で同時に満たす a・b（最小二乗）。リンゴの組が色の直接の効き目を 0 に近づける。
  4. 質問に答える: 「赤いトマトを緑に塗る」= 赤いトマトのグレインから始め、色だけを変える（介入）。予想される味の変化 = Σ_c 原因らしさ(c) × 効き目(c) × Δc。
       効き目(c) = 原因らしい組での Δ味 / Δc の平均。塗るのは色だけなので Δ色 = 1、Δ熟し具合 = 0。
     比べ: 相関だけの答え = 「色が緑のトマト」に最も近いグレイン（緑のトマト）の味をそのまま答える。
結果の置き場: results/exp63.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp63_correlation_to_cause.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grain import Grain, tv

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
VIEWS = {"色": ["赤", "緑", "黄", "白"], "味": ["甘い", "酸っぱい", "渋い"], "熟し具合": ["完熟", "未熟"], "種類": ["リンゴ", "トマト", "イチゴ", "バナナ"]}
VAL = [x for v in VIEWS.values() for x in v]; VN = list(VIEWS)
TABLE = {
    "赤いリンゴ": {"色": {"赤": 1}, "味": {"甘い": .9, "酸っぱい": .1}, "熟し具合": {"完熟": 1}, "種類": {"リンゴ": 1}},
    "青いリンゴ": {"色": {"緑": 1}, "味": {"甘い": .85, "酸っぱい": .15}, "熟し具合": {"完熟": 1}, "種類": {"リンゴ": 1}},     # 緑でも完熟の品種（王林など）
    "赤いトマト": {"色": {"赤": 1}, "味": {"甘い": .5, "酸っぱい": .5}, "熟し具合": {"完熟": 1}, "種類": {"トマト": 1}},
    "緑のトマト": {"色": {"緑": 1}, "味": {"甘い": .05, "酸っぱい": .8, "渋い": .15}, "熟し具合": {"未熟": 1}, "種類": {"トマト": 1}},
    "赤いイチゴ": {"色": {"赤": 1}, "味": {"甘い": .8, "酸っぱい": .2}, "熟し具合": {"完熟": 1}, "種類": {"イチゴ": 1}},
    "白いイチゴ": {"色": {"白": 1}, "味": {"甘い": .1, "酸っぱい": .9}, "熟し具合": {"未熟": 1}, "種類": {"イチゴ": 1}},
    "黄色いバナナ": {"色": {"黄": 1}, "味": {"甘い": 1}, "熟し具合": {"完熟": 1}, "種類": {"バナナ": 1}},
    "青いバナナ": {"色": {"緑": 1}, "味": {"甘い": .1, "渋い": .9}, "熟し具合": {"未熟": 1}, "種類": {"バナナ": 1}},
}
PAIRS = [("赤いリンゴ", "青いリンゴ"), ("赤いトマト", "緑のトマト"), ("赤いイチゴ", "白いイチゴ"), ("黄色いバナナ", "青いバナナ")]


def target(o):
    t = torch.zeros(len(VN), len(VAL))
    for i, v in enumerate(VN):
        for x, p in TABLE[o][v].items():
            t[i, VAL.index(x)] = p
    return t


def view_frac(W, v):
    cols = [VAL.index(x) for x in VIEWS[v]]; w = W[VN.index(v), cols]
    return w / w.sum()


res = []
for seed in range(3):
    G = {o: Grain(len(VN), len(VAL), seed=seed * 100 + k) for k, o in enumerate(TABLE)}
    fit = {o: G[o].fit(target(o), steps=800)[-1] for o in TABLE}
    A = {o: G[o].answer() for o in TABLE}
    # 1. 差分（明るさの引き算）
    D = {p: {v: float((view_frac(A[p[0]], v) - view_frac(A[p[1]], v)).abs().sum() / 2) for v in VN} for p in PAIRS}
    # 2. 原因らしさ（味の差と一緒に変わるか）
    cause = {}
    for c in ["色", "熟し具合"]:
        num = sum(D[p][c] * D[p]["味"] for p in PAIRS); den = math.sqrt(sum(D[p][c] ** 2 for p in PAIRS) * sum(D[p]["味"] ** 2 for p in PAIRS))
        cause[c] = num / den
    only_tomato = {c: D[("赤いトマト", "緑のトマト")][c] for c in ["色", "熟し具合"]}               # 組が 1 つだと区別がつかない
    # 3. 不確実性: 観点 c の差に対する味の差の、組ごとのばらつき（大きいほど「c → 味」は当てにならない）
    unc = {}
    for c in ["色", "熟し具合"]:
        r = [D[p]["味"] / D[p][c] for p in PAIRS if D[p][c] > 0.3]
        m = sum(r) / len(r); unc[c] = math.sqrt(sum((x - m) ** 2 for x in r) / len(r)) / max(m, 1e-9)
    # 4. 質問「赤いトマトを緑に塗ったら酸っぱくなるか？」
    eff = {c: sum(D[p]["味"] / max(D[p][c], 1e-9) for p in PAIRS if D[p][c] > 0.3) / max(1, sum(1 for p in PAIRS if D[p][c] > 0.3)) for c in ["色", "熟し具合"]}
    w = {c: max(cause[c], 0) ** 4 for c in cause}; s = sum(w.values()); w = {c: x / s for c, x in w.items()}   # 原因らしさを鋭く（4 乗、実験50 の非線形）
    paint = {"色": 1.0, "熟し具合": 0.0}                                                          # 塗る = 色だけを変える介入
    change = sum(w[c] * eff[c] * paint[c] for c in w)                                           # 予想される味の変化（0〜1）
    # 4b. 打ち消し合いを考える形: Δ味 ≈ a·Δ色 + b·Δ熟し具合 を、すべての組で同時に満たす a・b（最小二乗）。
    #     リンゴの組（色だけ変わって味は変わらない）が、色の直接の効き目 a を 0 に近づける（「説明し尽くす」= 交絡の除去）
    X = torch.tensor([[D[p]["色"], D[p]["熟し具合"]] for p in PAIRS]); y = torch.tensor([D[p]["味"] for p in PAIRS])
    ab = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]; coef = {"色": float(ab[0]), "熟し具合": float(ab[1])}
    change2 = max(0.0, min(1.0, coef["色"] * paint["色"] + coef["熟し具合"] * paint["熟し具合"]))
    base = view_frac(A["赤いトマト"], "味"); green = view_frac(A["緑のトマト"], "味")
    pred_intervene = (1 - change) * base + change * green                                       # 変化が大きいほど、緑のトマトの味へ寄る
    pred_intervene2 = (1 - change2) * base + change2 * green
    row = dict(exp="63", seed=seed, fit=fit, diff={f"{p[0]}↔{p[1]}": d for p, d in D.items()}, cause=cause, cause_tomato_only=only_tomato, uncertainty=unc, effect=eff, weight=w, change=change,
               赤いトマトの味={x: float(base[i]) for i, x in enumerate(VIEWS["味"])},
               介入の答え={x: float(pred_intervene[i]) for i, x in enumerate(VIEWS["味"])}, coef=coef, change2=change2,
               介入の答え_最小二乗={x: float(pred_intervene2[i]) for i, x in enumerate(VIEWS["味"])},
               相関だけの答え={x: float(green[i]) for i, x in enumerate(VIEWS["味"])})
    res.append(row)
    with open("results/exp63.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(seed, "差分", {f"{p[0]}↔{p[1]}": {v: round(x, 2) for v, x in d.items() if v != "種類"} for p, d in D.items()}, flush=True)
    print(seed, "原因らしさ", {c: round(x, 2) for c, x in cause.items()}, "トマトだけだと", {c: round(x, 2) for c, x in only_tomato.items()},
          "不確実さ", {c: round(x, 2) for c, x in unc.items()}, "予想される味の変化", round(change, 3), flush=True)
    print(seed, "最小二乗の効き目", {c: round(x, 3) for c, x in coef.items()}, "塗ったら（最小二乗）", {k: round(v, 2) for k, v in row["介入の答え_最小二乗"].items()}, flush=True)
    print(seed, "塗ったら（介入）", {k: round(v, 2) for k, v in row["介入の答え"].items()}, "相関だけ", {k: round(v, 2) for k, v in row["相関だけの答え"].items()}, flush=True)
