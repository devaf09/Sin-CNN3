"""実験37: 融合しやすい板を作る — 変化を小さく保つ学習・両方を知る出発点から派生（ほかのプロジェクトの知見と突き合わせ）

着想: 2026-10-27 持ち主「融合については、他のプロジェクトの成果や研究結果から、原因と解決方法を推敲してもいいかも」。
      掲示板の知見: 1 本の光路への差分の足し算（共通の出発点 θ0 + ΔA + ΔB）は 2 次元では壊れる（SN2-Parts-2D 実験09・Metasequoia 実験09）。
      並列の光路を明るさで足す融合は、崩れの原因がすべて「どの技能の担当か」の取り違えで、担当なしの学習で直る（Metasequoia 実験10・12）。
      実験36 で、共通の出発点は位相の平均による融合を少しだけ良くし、学習が短いほど良かった（板が離れる前）。
      → 板を 1 枚にまとめたいなら、2 枚が離れないようにすればよいのでは。2 つの作り方を試す:
        罰つき: 共通の初期 A0 から、変化 |A − A0|² に罰 λ を付けて学習（λ = 0・0.01・0.1・1）
        派生: まず両方の平均の答えで 1 枚（A_mix）を学習し、そこからリンゴ・トマトを 300 回だけ学習（両方を知る出発点）
      融合 = 2 枚の位相の平均 1 枚。
方法: 実験34・36 の器（単語ごとの格子 1 KB、初期の強さ 0.3）。3 seed。
指標: 融合した 1 枚の目標（2 つの平均）との差、元の 2 枚それぞれの表との差（罰でどれだけ覚えが悪くなるか）、変化の大きさ。
結果の置き場: results/exp37.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp37_fusion_friendly.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--lams", default="0,0.01,0.1,1")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--init", type=float, default=0.3)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp37.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp34_fusion.py"), encoding="utf-8").read().split("\nres = dict(exp=")[0]
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)
field, spec, grid, target_of, probe, Fw = ns["field"], ns["spec"], ns["grid"], ns["target_of"], ns["probe"], ns["a"].Fw
dist = lambda W, t: float((W / W.sum(1, keepdim=True) - t).abs().sum(1).mean() / 2)
out_of = lambda A: spec(field(torch.exp(1j * grid(A)), probe))


def train(A, tgt, steps, A0=None, lam=0.0):
    opt = torch.optim.Adam([A], lr=0.01); q = torch.cat([tgt, torch.zeros(len(tgt), 1)], 1)
    for step in range(steps):
        W = out_of(A); p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).mean() + (lam * ((A - A0) ** 2).sum() if lam else 0)
        opt.zero_grad(); loss.backward(); opt.step()


res = dict(exp="37", args=vars(a), rows=[])
tA, tB = target_of("リンゴ"), target_of("トマト"); mix = (tA + tB) / 2
for seed in range(a.seeds):
    g = torch.Generator().manual_seed(seed); A0 = a.init * torch.randn(4, Fw, Fw, generator=g)
    cases = [(f"罰つき λ={lam}", "pen", float(lam)) for lam in a.lams.split(",")] + [("派生（両方を知る出発点から 300 回）", "derive", 0.0)]
    for name, kind, lam in cases:
        t0 = time.time()
        if kind == "pen":
            AA, AB = torch.nn.Parameter(A0.clone()), torch.nn.Parameter(A0.clone())
            train(AA, tA, a.steps, A0, lam); train(AB, tB, a.steps, A0, lam); base = A0
        else:
            Am = torch.nn.Parameter(A0.clone()); train(Am, mix, a.steps); base = Am.detach().clone()
            AA, AB = torch.nn.Parameter(base.clone()), torch.nn.Parameter(base.clone()); train(AA, tA, 300); train(AB, tB, 300)
        with torch.no_grad():
            r = dict(case=name, seed=seed, fuse=dist(out_of((AA + AB) / 2), mix), tvA=dist(out_of(AA), tA), tvB=dist(out_of(AB), tB),
                     change=float(((AA - base).norm() + (AB - base).norm()) / 2 / base.norm()), sec=time.time() - t0)
        res["rows"].append(r)
        print(f"seed {seed} {name:<24} 融合 1 枚の差 {r['fuse']:.3f}  元の 2 枚の表との差 {r['tvA']:.3f}・{r['tvB']:.3f}  変化 {r['change']:.2f}  {r['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
