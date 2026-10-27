"""実験44: 差分（引き算）— 2 つの単語の違いを光で取り出す / 融合した 1 枚から片方を引いて戻す

着想: 2026-10-28 持ち主「融合と差分は大きな柱だったから、可能なら今後も改良していこう。差分についてはどうかな、引き算の実験もおねがいね」。
      実験34 の差分（半周ずらして場を足す）は、答えの違う所ほど明るくなったが弱かった（相関 0.55）。実験37 で、両方を知る出発点から派生させると
      2 枚が同じ「谷」に入り、位相の平均で融合できた。同じ谷の中なら、引き算もできるのでは。
方法: 実験37 の作り（単語ごとの格子 1 KB、初期の強さ 0.3）。リンゴ・トマトを (1) 別々の初期値から、(2) 両方を知る 1 枚 Am から派生（300 回）で学習。
      A. 違いを取り出す（目標 = 2 つの答えの差の大きさ |リンゴ − トマト|、値 12 × 観点 5 の模様）:
         明るさの引き算 |P_リンゴ − P_トマト|（2 枚の出口を測って引く）/ 場の引き算（トマト側を半周ずらして足す）/ 位相の引き算 1 枚 exp(i·(G_リンゴ − G_トマト + G_Am))
      B. 融合から片方を引いて戻す（目標 = トマトの答え）:
         明るさ: 2 × 融合（明るさの平均）− P_リンゴ / 位相: 2·A_m − A_リンゴ（位相の引き算。別々のときは A_m = 2 枚の平均なので恒等的に A_トマトに戻り、意味がない。派生のときだけ意味がある）/
         学習で引く: 融合した 1 枚から、トマトの答えに向けて 100 回だけ学習
指標: A は目標の模様との相関、B はトマトの答えとの差（全変動距離）。3 seed。
結果の置き場: results/exp44.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp44_difference.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--init", type=float, default=0.3)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp44.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp37_fusion_friendly.py"), encoding="utf-8").read().split("\nres = dict(exp=")[0]
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)
field, spec, grid, target_of, probe, Fw, out_of, train, dist = (ns[k] for k in ("field", "spec", "grid", "target_of", "probe", "Fw", "out_of", "train", "dist"))
norm = lambda W: W / W.sum(1, keepdim=True)
corr = lambda x, y: float(np.corrcoef(np.asarray(x).ravel(), np.asarray(y).ravel())[0, 1])
tA, tB = target_of("リンゴ"), target_of("トマト"); mix = (tA + tB) / 2; dtrue = (tA - tB).abs()
res = dict(exp="44", args=vars(a), rows=[])
for seed in range(a.seeds):
    g = torch.Generator().manual_seed(seed); A0 = a.init * torch.randn(4, Fw, Fw, generator=g); A0b = a.init * torch.randn(4, Fw, Fw, generator=g)
    for mode in ("別々", "派生"):
        t0 = time.time()
        if mode == "別々":
            AA, AB = torch.nn.Parameter(A0.clone()), torch.nn.Parameter(A0b.clone()); train(AA, tA, a.steps); train(AB, tB, a.steps)
            Am = ((AA + AB) / 2).detach()
        else:
            Pm = torch.nn.Parameter(A0.clone()); train(Pm, mix, a.steps); Am = Pm.detach().clone()
            AA, AB = torch.nn.Parameter(Am.clone()), torch.nn.Parameter(Am.clone()); train(AA, tA, 300); train(AB, tB, 300)
        with torch.no_grad():
            PA, PB = out_of(AA), out_of(AB); fA = field(torch.exp(1j * grid(AA)), probe); fB = field(torch.exp(1j * grid(AB)), probe)
            A_res = {"明るさの引き算": corr((norm(PA) - norm(PB)).abs(), dtrue),
                     "場の引き算（半周ずらす）": corr(norm(spec((fA - fB) / math.sqrt(2))), dtrue),
                     "位相の引き算 1 枚": corr(norm(out_of(AA - AB + Am)), dtrue)}
            fused = (norm(PA) + norm(PB)) / 2
            B_res = {"明るさ: 2 × 融合 − リンゴ": dist((2 * fused - norm(PA)).clamp(min=0), tB),
                     "位相: 2·A_m − A_リンゴ": dist(out_of(2 * Am - AA), tB)}
        Pf = torch.nn.Parameter(Am.clone()); train(Pf, tB, 100)                          # 融合した 1 枚から、学習で引く
        with torch.no_grad():
            B_res["学習で引く（100 回）"] = dist(out_of(Pf), tB); B_res["比べ: 融合した 1 枚そのまま"] = dist(out_of(Am), tB)
        r = dict(seed=seed, mode=mode, A=A_res, B=B_res, sec=time.time() - t0); res["rows"].append(r)
        print(f"seed {seed} {mode}: 違いの取り出し（相関）" + " ".join(f"{k} {v:.2f}" for k, v in A_res.items()) +
              "  | トマトに戻す（差）" + " ".join(f"{k} {v:.3f}" for k, v in B_res.items()) + f"  {r['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
