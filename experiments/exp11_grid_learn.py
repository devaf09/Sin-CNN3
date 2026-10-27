"""実験11: 周波数格子の学習器を学習させる（曲がる処理なし）— 何組の「入力 → 答え」を覚えられるか、表と比べる

着想: 実験09 の器（持ち主の案）。曲がる処理なしなら全体は 1 枚の表と同じはず。では学習で、表（100 × 100 の複素数の行列）と同じだけ覚えられるか。
      あわせて、格子の作り方（持ち主の案: 縦 sin × 横 cos / 私の案: 縦横とも sin と cos の 4 組）で差が出るかを見る。
方法: 入力 = でたらめな複素数の列（縦 100、P 個）、答え = 10 クラスのどれか（でたらめに割り当て = 覚えるしかない課題）。
      光を左から右へ 100 列通し（実験09 と同じ: 位相の遅れ → 1 画素の回折、縦 2 倍にゼロ詰め）、右端の列の 10 個の窓（6 画素ずつ）の明るさで答える。
      損失 = 窓の明るさの割合の交差エントロピー。Adam（学習率 --lr、既定 0.01、300 回）。周波数は 1〜F（F = 16）。
      比較相手: 表 = 100 × 100 の複素数の行列を入力に掛け、同じ窓で読む（同じ学習）。偶然 = 10%。
指標: 覚えた割合（訓練の正解率）、学習する数、時間。3 seed の平均 ± 標準誤差。
結果の置き場: results/exp11.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp11_grid_learn.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100, help="縦の画素数 = 列の数")
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--lr", type=float, default=0.01, help="0.05 だと sin と cos の格子が学べなかった（学び 11）")
ap.add_argument("--init", type=float, default=0.1, help="格子の強さの初期値の大きさ")
ap.add_argument("--kinds", default="all", help="all か、名前の先頭（例: 縦横）")
ap.add_argument("--Ps", default="10,30,100,300")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp11.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)                       # CPU は 2 本だけ使う（PC の負荷を抑える）
N, F, L, C = a.N, a.F, 2 * a.N, 10
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)   # (F, N)
H = torch.exp(1j * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))           # 1 列進む回折
win = torch.zeros(C, N); cen = torch.linspace(8, N - 9, C).long()
for c in range(C):
    win[c, cen[c] - 3:cen[c] + 3] = 1                                                   # 右端の窓 10 個（6 画素ずつ）


def grid(kind, A):
    if kind == "持ち主の案（縦 sin × 横 cos）":
        return sy.T @ A[0] @ cy
    return sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))   # 縦横とも sin と cos


def forward(kind, A, u0):
    if kind == "表（100 × 100 の行列）":
        return u0 @ A.T
    G = grid(kind, A)
    u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)            # 格子の列を通す
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")   # 1 列ぶん回折
    return u[:, :N]


res = dict(exp="11", args=vars(a), rows=[])
for kind in [k for k in ("持ち主の案（縦 sin × 横 cos）", "縦横とも sin と cos", "表（100 × 100 の行列）") if a.kinds == "all" or k.startswith(a.kinds)]:
    for P in [int(p) for p in a.Ps.split(",")]:
        accs, t0 = [], time.time()
        for seed in range(a.seeds):
            g = torch.Generator().manual_seed(seed)
            u0 = torch.complex(torch.randn(P, N, generator=g), torch.randn(P, N, generator=g)) / math.sqrt(2 * N)
            y = torch.randint(0, C, (P,), generator=g)
            if kind.startswith("表"):
                A = torch.nn.Parameter(torch.complex(torch.randn(N, N, generator=g), torch.randn(N, N, generator=g)) / math.sqrt(2 * N))
            else:
                A = torch.nn.Parameter(a.init * torch.randn(1 if kind.startswith("持ち主") else 4, F, F, generator=g))
            opt = torch.optim.Adam([A], lr=a.lr)
            for step in range(a.steps):
                I = forward(kind, A, u0).abs() ** 2 @ win.T                             # 窓の明るさ (P, C)
                loss = torch.nn.functional.cross_entropy(torch.log(I / I.sum(1, keepdim=True) + 1e-9), y)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                accs.append(float(((forward(kind, A, u0).abs() ** 2 @ win.T).argmax(1) == y).float().mean()))
        n_par = A.numel() * (2 if A.is_complex() else 1)
        r = dict(kind=kind, P=P, acc=(float(np.mean(accs)), float(np.std(accs, ddof=1) / math.sqrt(len(accs)))), params=n_par,
                 sec_per_seed=(time.time() - t0) / a.seeds)
        res["rows"].append(r)
        print(f"{kind:<22} P={P:>3}  覚えた割合 {100 * r['acc'][0]:5.1f}±{100 * r['acc'][1]:.1f}%  学習する数 {n_par}  {r['sec_per_seed']:.0f} 秒/seed", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
