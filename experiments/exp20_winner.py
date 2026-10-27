"""実験20: 強い道だけを残す — 2 つの記号が重なった光から、明るい方だけを通せるか（曲がる処理の使い道）

着想: 実験17 で、枝分かれは重ね合わせのまま同時に進み、光は薄く分かれていった。考え（推論）では、強い候補だけを残して弱い候補を消したい。
      これは表計算（線形）ではできないはず: 線形の器では、出力の B と C の明るさの比は入力の比を写すだけで、「どちらが明るいか」で態度を変えられない。
      曲がる処理（持ち主の案）なら、明るい方だけを通す「勝者がすべて取る」ができるかを確かめる。
方法: 記号は実験16 の周波数の波（11 個のうち 6 個 A〜F を使う）。入力 = 2 つの記号の重ね合わせ（明るさの比は 勝者 55〜90% : 敗者、位相はでたらめ）。
      答え = 勝者の記号だけが出口で光ること（同じ記号の周波数で読む）。器は実験16 と同じ（列の間 64 画素、周波数 1〜16、縦横とも sin と cos）。
      曲がる処理（1 色、列ごと）: なし / カー効果 u·e^(iγ|u|²)（γ = 30・100・300。光の合計は 1 で、1 画素あたりの明るさは 0.01 ほど）/ 飽和 u/√(1 + |u|²/s)（s = 0.01・0.003）。
      訓練 400・試験 400、Adam（0.01）600 回、3 seed。
指標: 鋭さの増え方 = 出口での勝者の割合（勝者 ÷（勝者 + 敗者）の明るさ）− 入口での勝者の割合。線形なら平均 0 以下のはず。勝者が一番明るい割合も。
結果の置き場: results/exp20.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp20_winner.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--kinds", default="なし,カー効果 γ=30,カー効果 γ=100,カー効果 γ=300,飽和 s=0.01,飽和 s=0.003")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp20.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L, K = a.N, a.F, 2 * a.N, 6
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
freq = torch.tensor([-10 + 2 * i for i in range(K)]); bins = freq % N
waves = (torch.exp(1j * 2 * math.pi * freq[:, None].float() * pos / N) / math.sqrt(N)).to(torch.complex64)   # (K, N)


def data(n, g):
    pair = torch.stack([torch.randperm(K, generator=g)[:2] for _ in range(n)])         # [勝者, 敗者]
    share = 0.55 + 0.35 * torch.rand(n, generator=g)                                    # 勝者の明るさの割合
    amp = torch.stack([share.sqrt(), (1 - share).sqrt()], 1) * torch.exp(2j * math.pi * torch.rand(n, 2, generator=g))
    return (amp[:, :, None] * waves[pair]).sum(1), pair, share


def run(kind, A, u0):
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
    u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
        if kind.startswith("カー効果"):
            u = u * torch.exp(1j * float(kind.split("=")[1]) * u.abs() ** 2)
        elif kind.startswith("飽和"):
            u = u / torch.sqrt(1 + u.abs() ** 2 / float(kind.split("=")[1]))
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs()[:, bins] ** 2       # 記号ごとの明るさ (n, K)


res = dict(exp="20", args=vars(a), rows=[])
for kind in a.kinds.split(","):
    gains, wins, t0 = [], [], time.time()
    for seed in range(a.seeds):
        g = torch.Generator().manual_seed(seed)
        (ux, px, sx), (vx, pv, sv) = data(400, g), data(400, g)
        A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
        for step in range(a.steps):
            W = run(kind, A, ux)
            p = torch.cat([W, (W.new_ones(len(W)) * (ux.abs() ** 2).sum(1) - W.sum(1)).clamp(min=1e-9)[:, None]], 1) / (ux.abs() ** 2).sum(1, keepdim=True)
            loss = -torch.log(p[torch.arange(len(W)), px[:, 0]] + 1e-9).mean()              # 勝者の記号に光を集める
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W = run(kind, A, vx); w, l = W[torch.arange(400), pv[:, 0]], W[torch.arange(400), pv[:, 1]]
            gains.append(float((w / (w + l) - sv).mean())); wins.append(float((W.argmax(1) == pv[:, 0]).float().mean()))
    m = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))))
    r = dict(kind=kind, gain=m(gains), win=m(wins), gain_each=gains, sec_per_seed=(time.time() - t0) / a.seeds); res["rows"].append(r)
    print(f"{kind:<12} 鋭さの増え方 {100 * r['gain'][0]:+5.1f}±{100 * r['gain'][1]:.1f} 点（各 seed {[round(100 * v, 1) for v in gains]}）  勝者が一番明るい {100 * r['win'][0]:5.1f}%  {r['sec_per_seed']:.0f} 秒/seed", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
