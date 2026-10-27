"""実験19: カー効果は光をどう曲げ合って「どちらか片方だけ」を解いているか — 4 通りの入力の光の 2 次元マップ

着想: 実験14 で、色を分けた「どちらか片方だけ」をカー効果ありの光の格子が 100% 解いた。しくみの見立て（ある色の通り道がほかの色の明るさで曲がる）を、
      光の 2 次元マップで確かめる。同じ学習済みの格子で、カー効果を切った場合（γ = 0）も並べる。
方法: 実験14 と同じ器・課題・学習（γ = 2、seed 0、400 回）。学習後、4 通りの入力（なし・A だけ・B だけ・A と B）の位相をそろえた代表例（揺れなし）を通し、
      色ごとの光の 2 次元マップと、右端の窓の明るさを記録する。カー効果を切った版（γ = 0、同じ格子）も同じように通す。
結果の置き場: results/exp19.txt、マップは results/exp19_maps.npz。図は tools/fig_exp19.py → docs/images/exp19_kerr_maps.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp19_kerr_maps.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp19.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
bump = lambda c: torch.exp(-((pos - c) / 4) ** 2)
win = torch.zeros(2, N); win[0, 20:40] = 1; win[1, 60:80] = 1
shape = torch.stack([bump(30), bump(70), bump(50)]).to(torch.complex64)


def data(n, g):
    ab = torch.randint(0, 2, (n, 2), generator=g).float()
    amp = (1 + 0.1 * torch.randn(n, 3, generator=g)) * torch.cat([ab, torch.ones(n, 1)], 1)
    return (amp * torch.exp(2j * math.pi * torch.rand(n, 3, generator=g)))[:, :, None] * shape[None], (ab[:, 0] != ab[:, 1]).long()


def run(A, u0, gamma, keep=False):
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
    u = torch.nn.functional.pad(u0, (0, L - N)); maps = []
    for col in range(N):
        u = torch.cat([u[..., :N] * torch.exp(1j * G[:, col]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
        u = u * torch.exp(1j * gamma * (u.abs() ** 2).sum(1, keepdim=True))             # カー効果（3 色の明るさの合計で）
        if keep:
            maps.append(u[..., :N].abs() ** 2)
    return (u[..., :N].abs() ** 2).sum(1) @ win.T, (torch.stack(maps, -1) if keep else None)


g = torch.Generator().manual_seed(a.seed); torch.manual_seed(a.seed)
(ux, y), (vx, vy) = data(400, g), data(400, g)
A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); opt = torch.optim.Adam([A], lr=0.01); t0 = time.time()
for step in range(a.steps):
    I, _ = run(A, ux, a.gamma)
    loss = torch.nn.functional.cross_entropy(torch.log(I / I.sum(1, keepdim=True) + 1e-9), y)
    opt.zero_grad(); loss.backward(); opt.step()
cases = {"なし": (0, 0), "A だけ": (1, 0), "B だけ": (0, 1), "A と B": (1, 1)}
u4 = torch.stack([torch.tensor([p, q, 1.0])[:, None] * shape for p, q in cases.values()])   # 位相をそろえた代表例
res = dict(exp="19", args=vars(a), rows=[])
with torch.no_grad():
    acc = float((run(A, vx, a.gamma)[0].argmax(1) == vy).float().mean())
    acc0 = float((run(A, vx, 0.0)[0].argmax(1) == vy).float().mean())
    W, M = run(A, u4, a.gamma, keep=True); W0, M0 = run(A, u4, 0.0, keep=True)
for n, k in enumerate(cases):
    res["rows"].append(dict(case=k, win_kerr=W[n].tolist(), win_off=W0[n].tolist()))
    print(f"{k:<5} 窓の明るさ（答え 0, 答え 1）: カー効果あり {W[n].numpy().round(3).tolist()}  切った {W0[n].numpy().round(3).tolist()}")
print(f"試験の正解率: カー効果あり {100 * acc:.1f}%  同じ格子でカー効果を切ると {100 * acc0:.1f}%  （学習 {time.time() - t0:.0f} 秒）")
res.update(acc_kerr=acc, acc_off=acc0, maps="results/exp19_maps.npz", numpy=np.__version__, torch=torch.__version__)
np.savez_compressed("results/exp19_maps.npz", kerr=M.numpy().astype(np.float16), off=M0.numpy().astype(np.float16))
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
