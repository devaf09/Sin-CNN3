"""実験14: 干渉を使えない「どちらか片方だけ」— 色を分けると、曲がる処理だけが 75% の壁を越えられるか

着想: 実験12 で、曲がる処理なしでも明るさの 2 乗と干渉で 83.3% まで届くと分かり、曲がる処理そのものの効果が見えなかった。
      A・B・参照の光を別々の色（互いに干渉しない光）にすると、曲がる処理なしでは窓の明るさが A と B の足し算になり、上限は本当に 75% になる。
      曲がる処理が「全部の色の明るさの合計」で効く（飽和・カー効果は色をまたいで効く）なら、色どうしが影響し合い、75% を越えられるはず。
方法: 光を 3 色（A・B・参照）に分け、同じ格子・同じ回折を別々に通す（色ごとの違いは今回は無視）。曲がる処理は 3 色の明るさの合計 I で:
        飽和 u/√(1 + I/s)（s = 0.3・1・3）、カー効果 u·e^(iγI)（γ = 0.5・2）。窓では 3 色の明るさを足して読む。
      器は実験12 と同じ（100 列、列の間 64 画素、周波数 1〜16、縦横とも sin と cos、窓 2 個）。比較相手: 小さなニューラルネットワーク、偶然 50%、上限 75%。
      訓練 400・試験 400、Adam（学習率 0.01、400 回）、3 seed。
結果の置き場: results/exp14.txt。図は tools/fig_exp14.py → docs/images/exp14_xor_colors.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp14_xor_colors.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--kinds", default="all")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp14.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
bump = lambda c: torch.exp(-((pos - c) / 4) ** 2)
win = torch.zeros(2, N); win[0, 20:40] = 1; win[1, 60:80] = 1
KINDS = {"なし": None, "飽和 s=0.3": ("sat", 0.3), "飽和 s=1": ("sat", 1.0), "飽和 s=3": ("sat", 3.0),
         "カー効果 γ=0.5": ("kerr", 0.5), "カー効果 γ=2": ("kerr", 2.0), "ニューラルネットワーク": "mlp"}


def data(n, g):
    ab = torch.randint(0, 2, (n, 2), generator=g).float()
    amp = (1 + 0.1 * torch.randn(n, 3, generator=g)) * torch.cat([ab, torch.ones(n, 1)], 1)
    ph = torch.exp(2j * math.pi * torch.rand(n, 3, generator=g))
    u = (amp * ph)[:, :, None] * torch.stack([bump(30), bump(70), bump(50)]).to(torch.complex64)[None]   # (n, 色 3, N)
    return u, (ab[:, 0] != ab[:, 1]).long()


def forward(kind, A, u0):
    if KINDS[kind] == "mlp":
        return A((u0.abs() ** 2).sum(1))
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
    u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[..., :N] * torch.exp(1j * G[:, col]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
        if KINDS[kind] is not None:
            I = (u.abs() ** 2).sum(1, keepdim=True)                                    # 3 色の明るさの合計（色をまたいで効く）
            kk, v = KINDS[kind]
            u = u / torch.sqrt(1 + I / v) if kk == "sat" else u * torch.exp(1j * v * I)
    return (u[..., :N].abs() ** 2).sum(1) @ win.T                                     # 3 色の明るさを足して窓で読む


res = dict(exp="14", args=vars(a), rows=[])
for kind in [k for k in KINDS if a.kinds == "all" or k.startswith(a.kinds)]:
    tr, te, t0 = [], [], time.time()
    for seed in range(a.seeds):
        g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
        (ux, y), (vx, vy) = data(400, g), data(400, g)
        if KINDS[kind] == "mlp":
            A = torch.nn.Sequential(torch.nn.Linear(N, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)); params = list(A.parameters())
        else:
            A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); params = [A]
        opt = torch.optim.Adam(params, lr=0.01)
        logit = (lambda o: o) if KINDS[kind] == "mlp" else (lambda I: torch.log(I / I.sum(1, keepdim=True) + 1e-9))
        for step in range(a.steps):
            loss = torch.nn.functional.cross_entropy(logit(forward(kind, A, ux)), y)
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tr.append(float((forward(kind, A, ux).argmax(1) == y).float().mean())); te.append(float((forward(kind, A, vx).argmax(1) == vy).float().mean()))
    m = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))))
    r = dict(kind=kind, train=m(tr), test=m(te), test_each=te, sec_per_seed=(time.time() - t0) / a.seeds)
    res["rows"].append(r)
    print(f"{kind:<12} 試験 {100 * r['test'][0]:5.1f}±{100 * r['test'][1]:.1f}%（各 seed {[round(100 * v, 1) for v in te]}）  訓練 {100 * r['train'][0]:5.1f}%  {r['sec_per_seed']:.0f} 秒/seed", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
