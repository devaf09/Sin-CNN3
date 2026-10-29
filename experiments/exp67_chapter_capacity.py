"""実験67: 章板に何語まで重ねられるか — 重ねる語の数 × 板の厚さ（GPU）、と章板を重ねた単位板

着想: 実験66 で、1 つの 2D の器（板 16 枚）に入口の角度を変えて 32 語を重ねると、語ごとの器との差は小さかった（3.08 対 2.86 ビット）。
      持ち主の「語板 → 章板 → 単位板」に向けて、(1) 何語まで重ねられるか、(2) 板を厚くすればもっと重ねられるか、(3) 章板をさらに重ねて単位板にできるか、を調べる。
方法: 器は実験66 と同じ 2D（板 32 × 32、2D の角スペクトル法、出口 52 点）。会話の名詞（非公開）、5 観点、実験58 の決まり。
  (1)(2) 1 つの器に W 語（W = 32・64・128・256）、板 K 枚（K = 8・16・32）。語 w・観点 v の入口 = 傾いた平らな光（下の tilt）。
          入口の傾きは、番号 i = 語·5 + 観点 を 32 × 32 の傾きの格子に順に並べる（どの 2 つも直交）。格子は 1024 点なので、W ≤ 204 語まで全部直交、
          W = 256 では入りきらずに重なる（2D の板 32 × 32 で重ねられる上限を見る）。
  (3) 単位板の模型: 章板（1 つの器に 32 語）を 2 つ学び、明るさで足す（並べて足す）/ 2 つの章を 1 つの器（64 語）に重ねる、を比べる。
指標: 確かめ用の交差エントロピー（語ごとの器・数えた表と比べる）、値に届く光、学習の時間。
結果の置き場: results/exp67.txt
実行: python experiments/exp67_chapter_capacity.py --device cuda --Ws 32,64 --Ks 8,16,32 / --Ws 128,256 ...（experiments/sweeps/sweep67a.sh・b.sh）
"""
import argparse, json, math, os, runpy, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--Ws", default="32,64,128,256")
ap.add_argument("--Ks", default="8,16,32")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--device", default="cpu")
ap.add_argument("--unit", action="store_true", help="(3) 単位板の模型も行う")
ap.add_argument("--out", default="results/exp67.txt")
a = ap.parse_args()
sys.argv = [sys.argv[0], "--seeds", "0"]
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "exp56_goban_grow.py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import evaluate

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); dev = a.device
NTR, NTE, prior = ns["NTR"], ns["NTE"], ns["prior"]


def shift(T, step=9):
    out = T.clone()
    for v in range(T.shape[1]):
        out[:, v, [(r + step * v) % 47 for r in range(47)]] = T[:, v, :47]
    return out


T, TE = shift(NTR[:400]), shift(NTE[:400]); pr = prior(shift(NTR))
TG = (T + 5 * pr[None]) / (T.sum(2, keepdim=True) + 5); NV, NX = TG.shape[1], TG.shape[2]
N, DZ = 32, 32.0; L = 2 * N
fx = torch.fft.fftfreq(L); KX, KY = torch.meshgrid(fx, fx, indexing="ij")
H = torch.exp(1j * DZ * torch.sqrt(((2 * math.pi) ** 2 - (2 * math.pi) ** 2 * (KX ** 2 + KY ** 2)).clamp(min=0))).to(dev)
SI = torch.tensor([8 + i // 7 for i in range(NX)], device=dev); SJ = torch.tensor([8 + i % 7 for i in range(NX)], device=dev)
yy, xx = torch.meshgrid(torch.arange(N, dtype=torch.float32), torch.arange(N, dtype=torch.float32), indexing="ij")
wave2d = lambda a_, b_: (torch.exp(1j * 2 * math.pi * (a_ * xx + b_ * yy) / N) / N).to(torch.complex64)


def tilt(w, v, W):
    """語 w・観点 v の入口の傾き。番号 i = w·5 + v を、32 × 32 の傾きの格子に順に並べる（横 = i mod 32、縦 = i div 32）。
    格子の点どうしは直交するので、W·5 ≤ 1024（W ≤ 204 語）なら全部の入口が直交。W = 256 では入りきらず、回り込んで重なる（器の上限を見る）。"""
    i = w * NV + v
    return ((i % N) - N // 2, (i // N) % N - N // 2)


def device(P, U):
    """P: (K, N, N) の共通の器、U: (語, 観点, N, N) の入口。出口の点の明るさ (語, 観点, 値)。"""
    u = torch.nn.functional.pad(U, (0, N, 0, N))
    for k in range(P.shape[0]):
        u = torch.cat([torch.cat([u[..., :N, :N] * torch.exp(1j * P[k]), u[..., :N, N:]], -1), u[..., N:, :]], -2)
        u = torch.fft.ifft2(torch.fft.fft2(u, norm="ortho") * H, norm="ortho")
    return torch.fft.fft2(u[..., :N, :N], norm="ortho").abs()[..., SI, SJ] ** 2


def fit(K, U, tg, seed):
    g = torch.Generator().manual_seed(seed)
    P = torch.nn.Parameter((math.pi * (torch.rand(K, N, N, generator=g) < 0.5).float()).to(dev)); opt = torch.optim.Adam([P], lr=0.03)
    q = torch.cat([tg, torch.zeros(*tg.shape[:-1], 1)], -1).to(dev); B = 64                     # 語を 64 ずつに分けて計算（メモリ節約、勾配は足し合わせる）
    for s in range(a.steps):
        opt.zero_grad()
        for i in range(0, U.shape[0], B):
            W = device(P, U[i:i + B]); p = torch.cat([W, (1 - W.sum(-1, keepdim=True)).clamp(min=1e-9)], -1)
            loss = -(q[i:i + B] * torch.log(p + 1e-9)).sum(-1).mean() * W.shape[0] / U.shape[0]; loss.backward()
        opt.step()
    with torch.no_grad():
        return P.detach(), torch.cat([device(P, U[i:i + B]) for i in range(0, U.shape[0], B)]).cpu()


pr2 = lambda W: 0.98 * W / W.sum(-1, keepdim=True).clamp(min=1e-12) + 0.02 / W.shape[-1]
for W in [int(x) for x in a.Ws.split(",")]:
    U = torch.stack([torch.stack([wave2d(*tilt(w, v, W)) for v in range(NV)]) for w in range(W)]).to(dev)
    for K in [int(x) for x in a.Ks.split(",")]:
        t0 = time.time(); P, Wout = fit(K, U, TG[:W], seed=W * 100 + K)
        ev = evaluate(pr2(Wout), TE[:W]); tab = evaluate(TG[:W], TE[:W])
        row = dict(exp="67", part="章板", W=W, K=K, bits=ev["bits"], table=tab["bits"], light=float(Wout.sum(-1).mean()), phases=K * N * N, sec=time.time() - t0, device=dev)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"章板 {W} 語・板 {K} 枚: 当たり {ev['bits']:.3f}（表 {tab['bits']:.3f}）光 {100 * row['light']:.1f}%  {row['sec']:.0f} 秒", flush=True)
if a.unit:
    K = 16; t0 = time.time()
    U32 = torch.stack([torch.stack([wave2d(*tilt(w, v, 32)) for v in range(NV)]) for w in range(32)]).to(dev)
    _, W1 = fit(K, U32, TG[:32], 1); _, W2 = fit(K, U32, TG[32:64], 2)                       # 章 1・章 2 を別々の章板に
    both = torch.cat([W1, W2]); ev_sep = evaluate(pr2(both), TE[:64])
    U64 = torch.stack([torch.stack([wave2d(*tilt(w, v, 64)) for v in range(NV)]) for w in range(64)]).to(dev)
    _, W12 = fit(K, U64, TG[:64], 3); ev_one = evaluate(pr2(W12), TE[:64])                    # 2 章を 1 つの器に
    # 単位板として読む: 章 1 の語と章 2 の語を同時に（干渉しない光源）入れたとき、答えは 2 語の平均になるか
    pairs = [(i, 32 + i) for i in range(32)]
    avg = torch.stack([(pr2(both[i]) + pr2(both[j])) / 2 for i, j in pairs]); mix = torch.stack([pr2((both[i] + both[j]) / 2) for i, j in pairs])
    row = dict(exp="67", part="単位板", K=K, 章板2つ=ev_sep["bits"], 一つの器に2章=ev_one["bits"], 表=evaluate(TG[:64], TE[:64])["bits"],
               章をまたぐ融合の差=float((mix - avg).abs().sum(-1).mean() / 2), sec=time.time() - t0, device=dev)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("単位板:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
