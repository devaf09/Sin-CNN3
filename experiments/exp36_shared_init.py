"""実験36: 同じ初期の格子から学習すると、位相を混ぜる融合ができるか（共通の出発点）— と「万能初期乱数」の下見

着想: 2026-10-27 持ち主「初期の格子は学習後もなんとなく形が残ってる気がする。ある種の最適化が逆に可能なのでは（万能初期乱数、
      もしくは光路の分岐が容易に起きる）。最終的に融合に影響があるのかな」。実験34 で位相の平均が壊れたのは、リンゴとトマトが別々の初期値から
      学習したからかもしれない。Sin-CNN2 の実験23 では、共通の出発点から派生させると足し算で 1 本の光路に融合できた。
方法: 実験34 と同じ器（単語ごとの格子 1 KB、入口は「何か」× 観点、実験21 の表）。
        共通: リンゴとトマトを同じ初期の格子 A0 から学習。 別々: 別の初期値から学習（実験34 と同じ）。
      融合: 平均 = (A_リンゴ + A_トマト)/2（共通なら A0 + (変化_リンゴ + 変化_トマト)/2）、変化の和 = A0 + 変化_リンゴ + 変化_トマト（共通のときだけ）、
            比べ: 2 枚に分けて明るさを足す。
      初期の格子の強さ s = 0.03・0.1・0.3 も変える（万能初期乱数の下見: 覚えやすさ = 表との差が 0.15 を切るまでの学習の回数）。
指標: 目標（2 つの平均）との差、値に届いた光、学習による格子の変化の大きさ（|A − A0| / |A0|）、表との差が 0.15 を切った回数。3 seed。
結果の置き場: results/exp36.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp36_shared_init.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--scales", default="0.03,0.1,0.3")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp36.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp34_fusion.py"), encoding="utf-8").read().split("\nres = dict(exp=")[0]   # 行の頭の res = dict(exp= で切る
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)                   # 実験34 の器・表・関数
field, spec, grid, target_of, probe, Fw = ns["field"], ns["spec"], ns["grid"], ns["target_of"], ns["probe"], ns["a"].Fw
dist = lambda W, t: float((W / W.sum(1, keepdim=True) - t).abs().sum(1).mean() / 2)


def train(A, tgt, steps):
    """学習し、表との差が 0.15 を切った回数を返す（切らなければ steps）。"""
    opt = torch.optim.Adam([A], lr=0.01); hit = steps
    q = torch.cat([tgt, torch.zeros(len(tgt), 1)], 1)
    for step in range(steps):
        W = spec(field(torch.exp(1j * grid(A)), probe)); p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        if hit == steps and step % 25 == 0 and dist(W.detach(), tgt) < 0.15:
            hit = step
    return hit


res = dict(exp="36", args=vars(a), rows=[])
for s in [float(x) for x in a.scales.split(",")]:
    for seed in range(a.seeds):
        t0 = time.time(); g = torch.Generator().manual_seed(seed); row = dict(scale=s, seed=seed)
        A0 = s * torch.randn(4, Fw, Fw, generator=g); A0b = s * torch.randn(4, Fw, Fw, generator=g)
        tA, tB = ns["target_of"]("リンゴ"), ns["target_of"]("トマト"); mix = (tA + tB) / 2
        for mode, (initA, initB) in (("共通", (A0, A0)), ("別々", (A0, A0b))):
            AA, AB = torch.nn.Parameter(initA.clone()), torch.nn.Parameter(initB.clone())
            hA, hB = train(AA, tA, a.steps), train(AB, tB, a.steps)
            with torch.no_grad():
                WA, WB = spec(field(torch.exp(1j * grid(AA)), probe)), spec(field(torch.exp(1j * grid(AB)), probe))
                cand = {"平均": (AA + AB) / 2}
                if mode == "共通":
                    cand["変化の和"] = A0 + (AA - A0) + (AB - A0)
                r = {k: dict(dist=dist(spec(field(torch.exp(1j * grid(v)), probe)), mix), light=float(spec(field(torch.exp(1j * grid(v)), probe)).sum(1).mean())) for k, v in cand.items()}
                r["明るさを足す"] = dict(dist=dist(0.5 * WA + 0.5 * WB, mix), light=float((0.5 * WA + 0.5 * WB).sum(1).mean()))
                row[mode] = dict(fuse=r, steps_to_015=[hA, hB], change=[float((AA - initA).norm() / initA.norm()), float((AB - initB).norm() / initB.norm())],
                                 tv=[dist(WA, tA), dist(WB, tB)])
        row["sec"] = time.time() - t0; res["rows"].append(row)
        c, d = row["共通"], row["別々"]
        print(f"初期の強さ {s} seed {seed}: 変化の大きさ {c['change'][0]:.2f}・{c['change'][1]:.2f}  0.15 を切った回 {c['steps_to_015']}  "
              f"融合（目標との差）共通: 平均 {c['fuse']['平均']['dist']:.3f}・変化の和 {c['fuse']['変化の和']['dist']:.3f}  別々: 平均 {d['fuse']['平均']['dist']:.3f}  "
              f"明るさを足す {c['fuse']['明るさを足す']['dist']:.3f}  {row['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
