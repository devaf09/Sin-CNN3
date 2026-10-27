"""実験12: 表では解けない課題「どちらか片方だけ」— 明るさが曲がる処理で解けるようになるか

着想: 実験09・11 の続き（持ち主の目標: 表計算からニューラルネットワークのような変化へ）。曲がる処理なしの器は表 1 枚と同じなので、
      表で解けない課題は解けないはず。曲がる処理（カー効果・飽和、持ち主の案）を入れると解けるかを、小さなニューラルネットワークと並べる。
課題: 縦 100 画素の光の列に、2 つの成分 A（30 番のあたりのふくらみ）・B（70 番のあたり）が「あり / なし」で入る（4 通り）。
      いつも点いている参照の光（50 番）もある。答えは「A と B のどちらか片方だけ点いている」か（XOR）。
      A・B・参照の光は、毎回でたらめな位相（光の波の山の位置）と、±10% の明るさの揺れを持つ。
      位相がでたらめなので、A と B の干渉（足し算の 2 乗で出る掛け算の項）は平均すると消え、明るさの読み出しは A・B の「あり / なし」の足し算になる。
      足し算では「どちらか片方だけ」は作れないので、曲がる処理なしの器と表は 75% 止まり（4 通りのうち 3 通りまで）のはず。
方法: 器は実験11 と同じ（100 列、周波数 1〜16、縦横とも sin と cos、右端の窓 2 個）。曲がる処理は列ごと: なし / カー効果 u·e^(iγ|u|²)（γ = 2）/ 飽和 u/√(1 + |u|²)。
      比較相手: 表（100 × 100 の複素数の行列 + 窓）、小さなニューラルネットワーク（入力の明るさ 100 → 16 → 2）、偶然 50%。
      訓練 400 個・試験 400 個、Adam（学習率 0.01、400 回）。3 seed の平均 ± 標準誤差。
結果の置き場: results/exp12.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp12_xor_bend.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--gamma", type=float, default=2.0)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--lr", type=float, default=0.01)
ap.add_argument("--cols", type=int, default=0, help="光が通る列の数（0 なら N）")
ap.add_argument("--kinds", default="all", help="all か、名前の先頭")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp12.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
bump = lambda c: torch.exp(-((pos - c) / 4) ** 2)                                     # ふくらみ（幅 4 画素）
win = torch.zeros(2, N); win[0, 20:40] = 1; win[1, 60:80] = 1                         # 右端の窓 2 個（答え 0 / 1）


def data(n, g):
    ab = torch.randint(0, 2, (n, 2), generator=g).float()
    amp = (1 + 0.1 * torch.randn(n, 3, generator=g)) * torch.cat([ab, torch.ones(n, 1)], 1)
    ph = torch.exp(2j * math.pi * torch.rand(n, 3, generator=g))                       # でたらめな位相
    u = (amp * ph) @ torch.stack([bump(30), bump(70), bump(50)]).to(torch.complex64)
    return u, (ab[:, 0] != ab[:, 1]).long()                                            # 答え = どちらか片方だけ


def forward(kind, A, u0):
    if kind == "表":
        return (u0 @ A.T).abs() ** 2 @ win.T
    if kind == "ニューラルネットワーク":
        return A(u0.abs() ** 2)
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
    u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(a.cols or N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
        if kind == "格子 + カー効果":
            u = u * torch.exp(1j * a.gamma * u.abs() ** 2)                             # 明るいほど遅れる
        elif kind == "格子 + 飽和":
            u = u / torch.sqrt(1 + u.abs() ** 2)                                       # 明るいほど通りにくい
    return u[:, :N].abs() ** 2 @ win.T


res = dict(exp="12", args=vars(a), rows=[])
for kind in [k for k in ("表", "格子（曲がる処理なし）", "格子 + カー効果", "格子 + 飽和", "ニューラルネットワーク") if a.kinds == "all" or k.startswith(a.kinds)]:
    tr, te, t0 = [], [], time.time()
    for seed in range(a.seeds):
        g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
        (ux, y), (vx, vy) = data(400, g), data(400, g)
        if kind == "表":
            A = torch.nn.Parameter(torch.complex(torch.randn(N, N, generator=g), torch.randn(N, N, generator=g)) / math.sqrt(2 * N)); params = [A]
        elif kind == "ニューラルネットワーク":
            A = torch.nn.Sequential(torch.nn.Linear(N, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2)); params = list(A.parameters())
        else:
            A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); params = [A]
        opt = torch.optim.Adam(params, lr=a.lr); losses = []
        logit = (lambda o: o) if kind == "ニューラルネットワーク" else (lambda I: torch.log(I / I.sum(1, keepdim=True) + 1e-9))
        for step in range(a.steps):
            loss = torch.nn.functional.cross_entropy(logit(forward(kind, A, ux)), y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
        with torch.no_grad():
            tr.append(float((forward(kind, A, ux).argmax(1) == y).float().mean())); te.append(float((forward(kind, A, vx).argmax(1) == vy).float().mean()))
    m = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))))
    r = dict(kind=kind, loss_first_last=[losses[0], losses[-1]], train=m(tr), test=m(te), params=sum(p.numel() * (2 if p.is_complex() else 1) for p in params), sec_per_seed=(time.time() - t0) / a.seeds)
    res["rows"].append(r)
    print(f"{kind:<14} 試験 {100 * r['test'][0]:5.1f}±{100 * r['test'][1]:.1f}%  訓練 {100 * r['train'][0]:5.1f}%  損失 {losses[0]:.3f}→{losses[-1]:.3f}  学習する数 {r['params']}  {r['sec_per_seed']:.0f} 秒/seed", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
