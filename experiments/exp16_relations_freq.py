"""実験16: 関係性概念を「混ざらない周波数の光」で — 記号を整数の周波数の波で表すと、少しのずらしで切り替えられるか

着想: 実験15 で、記号を細い光の点で表すと、縦の周波数を少しずらしても入力がほとんど変わらず（重なり 0.999）、第 2 段階が切り替わらなかった。
      持ち主の案「混ざらない周波数の光（素数的な）」: 記号を、縦 100 画素全体に広がる整数の周波数の波で表す。整数の周波数は 1 ずらすと
      ずらす前とぴったり重ならない（学び 06）ので、光を捨てずに切り替えられるはず。
方法: 記号 11 個（A B C D E F G H X Y Z）に、周波数 −10, −8, …, 10（100 画素で何周、偶数）を割り当て、入力 = e^(i2π·f·y/100)（明るさは一様）。
      出力は右端の列を FFT（norm="ortho"）して、記号の周波数の番号の明るさで読む（入力と同じ仕組み）。
      第 2 段階のずらし δ は奇数（1 など）なら、ずらした先がどの記号の周波数とも重ならない。δ = 0.5 は比べ（重なりが残る）。
      関係は実験15 と同じ（第 1 段階 A → B, C, D / B → C / D → A, E, F / X → Y, Z、ずらしたとき A → G, H / B → F / D → B, C / X → E, F, G）。
      器は実験15 と同じ（100 列、周波数 1〜16、縦横とも sin と cos、曲がる処理なし）。列の間 --dz 画素。Adam（0.01）1500 回、3 seed。
指標: 組がぴったり当たった割合、出口の光のうち答えの周波数に入った割合、入れた光のうち出口に届いた割合。
結果の置き場: results/exp16.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp16_relations_freq.py --shifts 0.5,1 --dz 64
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--shifts", default="0.5,1")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp16.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
SYM = list("ABCDEFGHXYZ"); freq = {s: -10 + 2 * i for i, s in enumerate(SYM)}        # 記号の周波数（偶数）
bins = torch.tensor([freq[s] % N for s in SYM])                                       # FFT の番号（負の周波数は後ろ）
REL0 = {"A": "BCD", "B": "C", "D": "AEF", "X": "YZ"}
REL1 = {"A": "GH", "B": "F", "D": "BC", "X": "EFG"}


def wave(s, delta=0.0):
    return (torch.exp(1j * 2 * math.pi * (freq[s] + delta) * pos / N) / math.sqrt(N)).to(torch.complex64)   # 明るさの合計 1


def run(A, u0):
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
    u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho")                        # 出口の列を周波数に分けて読む（明るさの合計は変えない）


def measure(Y, tgt):
    P = Y.abs() ** 2; Wi = P[:, bins]
    ok = [set(Wi[n].argsort(descending=True)[:len(t)].tolist()) == {SYM.index(c) for c in t} for n, t in enumerate(tgt)]
    frac = [float(Wi[n, [SYM.index(c) for c in t]].sum() / P[n].sum()) for n, t in enumerate(tgt)]
    return float(np.mean(ok)), float(np.mean(frac)), float(P.sum(1).mean()), Wi


res = dict(exp="16", args=vars(a), rows=[])
for delta in [float(d) for d in a.shifts.split(",")]:
    pairs = [(s, 0.0, t) for s, t in REL0.items()] + ([(s, delta, t) for s, t in REL1.items()] if delta > 0 else [])
    u0 = torch.stack([wave(s, d) for s, d, _ in pairs]); tgt = [t for _, _, t in pairs]
    ov = float(abs((torch.conj(wave("A")) * wave("A", delta)).sum()))                 # ずらす前と後の入力の重なり
    q = torch.zeros(len(pairs), len(SYM) + 1)
    for n, t in enumerate(tgt):
        for c in t:
            q[n, SYM.index(c)] = 1 / len(t)
    for seed in range(a.seeds):
        t0 = time.time(); g = torch.Generator().manual_seed(seed)
        A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
        with torch.no_grad():
            before = measure(run(A, u0), tgt)
        curve = []
        for step in range(a.steps):
            P = run(A, u0).abs() ** 2; Wi = P[:, bins]
            p = torch.cat([Wi, (1 - Wi.sum(1))[:, None].clamp(min=1e-9)], 1)              # 入れた光は 1。答え以外・出口の外は「はずれ」
            loss = -(q * torch.log(p + 1e-9)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0 or step == a.steps - 1:
                curve.append([step, float(loss.detach())])
        with torch.no_grad():
            after = measure(run(A, u0), tgt)
        r = dict(delta=delta, overlap=ov, seed=seed, before=before[:3], after=after[:3], win_after=after[3].tolist(), curve=curve, sec=time.time() - t0)
        res["rows"].append(r)
        print(f"δ={delta:<4} 重なり {ov:.3f} seed {seed}: 組がぴったり {100 * before[0]:5.1f} → {100 * after[0]:5.1f}%  答えの周波数の光 {100 * before[1]:4.1f} → {100 * after[1]:4.1f}%  "
              f"出口に届いた光 {100 * after[2]:4.1f}%  {r['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
