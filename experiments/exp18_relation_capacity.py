"""実験18: 関係をいくつ 1 つの器に入れられるか（容量）— 記号 24 個 × 文脈 4 つ

着想: 実験16・17 で、記号を周波数の波で表し、縦の周波数のずらし（文脈）で別の関係に切り替えられた。では関係を何組まで入れられるか。
方法: 記号 24 個に周波数 −48, −44, …, 44（4 の倍数）を割り当て、文脈 c = 0, 1, 2, 3 は縦の周波数を c だけずらす（ずらした先はどの記号とも重ならない）。
      （記号, 文脈）の組を P 個でたらめに選び、それぞれに 1〜3 個のでたらめな答えの記号を決める。器は実験16 と同じ（曲がる処理なし、列の間 64 画素）。
      格子の周波数 F = 16（学習する数 1024）と F = 32（4096）。比較相手: 表（100 × 100 の複素数の行列。入力は互いに重ならないので、96 組まで全部入るはず）。
      Adam（0.01）1500 回、3 seed。
指標: 組がぴったり当たった割合、出口の光のうち答えの周波数に入った割合。
結果の置き場: results/exp18.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp18_relation_capacity.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--Ps", default="8,16,32,64,96")
ap.add_argument("--kinds", default="格子 F=16,格子 F=32,表")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp18.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, L, S = a.N, 2 * a.N, 24
pos = torch.arange(N, dtype=torch.float32)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
freq = torch.tensor([-48 + 4 * i for i in range(S)]); bins = freq % N


def basis(F):
    fr = torch.arange(1, F + 1, dtype=torch.float32)
    return torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)


def run(kind, A, u0, B):
    if kind == "表":
        return torch.fft.fft(u0 @ A.T, n=N, dim=-1, norm="ortho")
    sy, cy = B
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
    u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho")


res = dict(exp="18", args=vars(a), rows=[])
for kind in a.kinds.split(","):
    for P in [int(p) for p in a.Ps.split(",")]:
        oks, fracs, t0 = [], [], time.time()
        for seed in range(a.seeds):
            g = torch.Generator().manual_seed(seed)
            pick = torch.randperm(S * 4, generator=g)[:P]; src, ctx = pick % S, pick // S       # （記号, 文脈）の組
            u0 = (torch.exp(1j * 2 * math.pi * (freq[src] + ctx)[:, None].float() * pos / N) / math.sqrt(N)).to(torch.complex64)
            q = torch.zeros(P, S + 1); tgt = []
            for n in range(P):
                t = torch.randperm(S, generator=g)[:int(torch.randint(1, 4, (1,), generator=g))]; tgt.append(set(t.tolist())); q[n, t] = 1 / len(t)
            if kind == "表":
                A = torch.nn.Parameter(torch.complex(torch.randn(N, N, generator=g), torch.randn(N, N, generator=g)) / math.sqrt(2 * N)); B = None
            else:
                F = int(kind.split("=")[1]); B = basis(F); A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g))
            opt = torch.optim.Adam([A], lr=0.01)
            for step in range(a.steps):
                Wi = run(kind, A, u0, B).abs()[:, bins] ** 2
                p = torch.cat([Wi, (1 - Wi.sum(1))[:, None].clamp(min=1e-9)], 1)
                loss = -(q * torch.log(p + 1e-9)).sum(1).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                P2 = run(kind, A, u0, B).abs() ** 2; Wi = P2[:, bins]
                oks.append(float(np.mean([set(Wi[n].argsort(descending=True)[:len(tgt[n])].tolist()) == tgt[n] for n in range(P)])))
                fracs.append(float(np.mean([float(Wi[n, list(tgt[n])].sum() / P2[n].sum()) for n in range(P)])))
        m = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))))
        r = dict(kind=kind, P=P, ok=m(oks), frac=m(fracs), sec_per_seed=(time.time() - t0) / a.seeds); res["rows"].append(r)
        print(f"{kind:<8} P={P:>3}  組がぴったり {100 * r['ok'][0]:5.1f}±{100 * r['ok'][1]:.1f}%  答えの光 {100 * r['frac'][0]:5.1f}%  {r['sec_per_seed']:.0f} 秒/seed", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
