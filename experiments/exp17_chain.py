"""実験17: 関係をつなげる — 出力の光をそのまま次の入力にして、A → B → C → … とたどれるか

着想: 実験16 で、記号を周波数の波で表すと入力と出力が同じ仕組みになった。だから出力の光を、そのまま同じ器にもう一度通せる（光を戻す）。
      関係をつないでたどること（A から B へ、B から C へ）は、「思考」の一番小さな形の 1 つ。
方法: 器・記号の表し方は実験16 と同じ（周波数の波、曲がる処理なし、列の間 64 画素）。2 つの関係の組で学習する（1 回ぶんの対応だけを教える）:
        一本道: A → B, B → C, C → D, D → E, E → F
        枝分かれ: A → B, C / B → D / C → E / D → F / E → G
      学習後、A を入れ、出口の光（右端の列の場）をそのまま次の入力にして 5 回通す。各回の出口で、記号の周波数ごとの明るさを読む。
      式の予想（光の器は重ね合わせが成り立つ）: 枝分かれでは、2 回目に B と C の先（D と E）が同時に光る。光は回ごとに（答えに届く割合）倍ずつ減る。
指標: 各回で一番明るい記号（枝分かれでは上位 2 つ）が予想どおりか、答えの記号に入った光（入れた光 = 1）。3 seed。
結果の置き場: results/exp17.txt。図は tools/fig_exp17.py → docs/images/exp17_chain.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp17_chain.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--hops", type=int, default=5)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp17.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
SYM = list("ABCDEFGHXYZ"); freq = {s: -10 + 2 * i for i, s in enumerate(SYM)}
bins = torch.tensor([freq[s] % N for s in SYM])
CHAINS = {"一本道": ({"A": "B", "B": "C", "C": "D", "D": "E", "E": "F"}, ["A", "B", "C", "D", "E", "F"]),
          "枝分かれ": ({"A": "BC", "B": "D", "C": "E", "D": "F", "E": "G"}, ["A", "BC", "DE", "FG", "", ""])}
wave = lambda s: (torch.exp(1j * 2 * math.pi * freq[s] * pos / N) / math.sqrt(N)).to(torch.complex64)


def run(A, u0):
    """光を器に 1 回通し、出口の列の場（周波数に分ける前）を返す。"""
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
    u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return u[:, :N]


spec = lambda u: torch.fft.fft(u, n=N, dim=-1, norm="ortho").abs() ** 2               # 記号の周波数ごとの明るさを読むため
res = dict(exp="17", args=vars(a), rows=[])
for name, (rel, expect) in CHAINS.items():
    src = list(rel); u0 = torch.stack([wave(s) for s in src])
    q = torch.zeros(len(src), len(SYM) + 1)
    for n, s in enumerate(src):
        for c in rel[s]:
            q[n, SYM.index(c)] = 1 / len(rel[s])
    for seed in range(a.seeds):
        t0 = time.time(); g = torch.Generator().manual_seed(seed)
        A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
        for step in range(a.steps):
            Wi = spec(run(A, u0))[:, bins]
            p = torch.cat([Wi, (1 - Wi.sum(1))[:, None].clamp(min=1e-9)], 1)
            loss = -(q * torch.log(p + 1e-9)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        hops, u = [], wave("A")[None]
        with torch.no_grad():
            for h in range(a.hops + 1):
                W = spec(u)[0, bins]; k = len(expect[h]) if h < len(expect) and expect[h] else 1
                top = "".join(SYM[i] for i in W.argsort(descending=True)[:k].tolist())
                ok = (set(top) == set(expect[h])) if h < len(expect) and expect[h] else None
                hops.append(dict(hop=h, sym_light=W.tolist(), top=top, expect=expect[h] if h < len(expect) else "", ok=ok,
                                 light_on_expect=float(sum(W[SYM.index(c)] for c in expect[h])) if h < len(expect) and expect[h] else None,
                                 light_total=float((u.abs() ** 2).sum())))
                u = run(A, u)                                                              # 出口の光をそのまま次の入力に
        r = dict(chain=name, seed=seed, hops=hops, sec=time.time() - t0); res["rows"].append(r)
        print(f"{name} seed {seed}: " + "  ".join(f"{x['hop']}回目 {x['top']}{'' if x['ok'] is None else ('○' if x['ok'] else '×')}"
              f"（答え {x['light_on_expect']:.2f}）" if x["light_on_expect"] is not None else f"{x['hop']}回目 {x['top']}" for x in hops), flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
