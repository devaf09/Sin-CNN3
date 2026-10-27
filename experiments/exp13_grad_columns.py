"""実験13: 光の格子の学習はどこで止まっているか — 列ごとの傾きの大きさと、窓に届く光

着想: 実験12 で、光の格子は「どちらか片方だけ」を学べなかった（実験11 の「でたらめな入力 → クラス」は学べた）。
      持ち主の疑問「勾配降下法は光式に合っていないのでは」（ツッコミ 3-4）に向けて、まず傾き（勾配）がどこでどれだけあるかを見る。
方法: 器は実験11・12 と同じ（100 列、周波数 1〜16、縦横とも sin と cos、学習する強さは初め 0.1 × でたらめ）。2 つの課題で、学習の前に:
        (1) 列ごとの傾きの大きさ: 位相の遅れ G[:, 列] を学習する数と見なしたときの |∂損失/∂G[:, 列]| の合計
        (2) 窓に届く光の割合（入力の明るさのうち、右端の窓に入った分）
      課題: 「覚える」= 実験11（でたらめな光の列 100 個 → 10 クラス、窓 10 個）、「片方だけ」= 実験12（ふくらみ A・B・参照 → 2 クラス、窓 2 個）。
指標: 列ごとの傾き（図）、窓に届く光の割合。seed 0。
結果の置き場: results/exp13.txt。図は tools/fig_exp13.py → docs/images/exp13_grad_columns.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp13_grad_columns.py
"""
import argparse, json, math, os, sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp13.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
g = torch.Generator().manual_seed(a.seed)
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
A = 0.1 * torch.randn(4, F, F, generator=g)
G0 = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))


def task(name):
    if name == "覚える（実験11）":
        u = torch.complex(torch.randn(100, N, generator=g), torch.randn(100, N, generator=g)) / math.sqrt(2 * N)
        y = torch.randint(0, 10, (100,), generator=g); win = torch.zeros(10, N); cen = torch.linspace(8, N - 9, 10).long()
        for c in range(10):
            win[c, cen[c] - 3:cen[c] + 3] = 1
        return u, y, win
    bump = lambda c: torch.exp(-((pos - c) / 4) ** 2)
    ab = torch.randint(0, 2, (400, 2), generator=g).float()
    amp = (1 + 0.1 * torch.randn(400, 3, generator=g)) * torch.cat([ab, torch.ones(400, 1)], 1)
    u = (amp * torch.exp(2j * math.pi * torch.rand(400, 3, generator=g))) @ torch.stack([bump(30), bump(70), bump(50)]).to(torch.complex64)
    win = torch.zeros(2, N); win[0, 20:40] = 1; win[1, 60:80] = 1
    return u, (ab[:, 0] != ab[:, 1]).long(), win


res = dict(exp="13", args=vars(a), rows=[])
for name in ("覚える（実験11）", "片方だけ（実験12）"):
    u0, y, win = task(name)
    G = G0.clone().requires_grad_(True)                                                # 位相の遅れを直接、学習する数と見なす
    u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    I = u[:, :N].abs() ** 2 @ win.T
    loss = torch.nn.functional.cross_entropy(torch.log(I / I.sum(1, keepdim=True) + 1e-9), y)
    loss.backward()
    gcol = G.grad.abs().sum(0)                                                          # 列ごとの傾きの大きさ
    frac = float((I.sum(1) / (u0.abs() ** 2).sum(1)).mean())                            # 窓に届く光の割合
    res["rows"].append(dict(task=name, grad_col=gcol.tolist(), window_frac=frac, loss=float(loss)))
    print(f"{name:<14} 損失 {float(loss):.3f}  窓に届く光 {100 * frac:.1f}%  傾き: 左 10 列 {float(gcol[:10].mean()):.2e}、"
          f"真ん中 {float(gcol[45:55].mean()):.2e}、右 10 列 {float(gcol[-10:].mean()):.2e}")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
