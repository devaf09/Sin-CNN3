"""実験48: 層より大きなレンズで、外に逃げた光も拾う（持ち主の案 4）

着想: 2026-10-28 持ち主「拡散してなくなる光にも意味を持たせて、層よりも大きなレンズで集光する」。
      これまでの器（実験42〜46）は、計算の窓を 2N にして板は真ん中の N 画素だけに置き、出口では板の幅 N の場だけをレンズ（FFT）に入れていた。
      板の外へ回折した光は、そのまま捨てていた。
方法: 実験46 の易しい課題（実験21 の表、縦 N = 128、画素ごとの板 20 列、列の間 64 画素、600 回、白色雑音 ±π から、3 seed）。
  レンズ N  : これまでどおり、出口の幅 N の場を FFT（値 k = 番号 k）
  レンズ 2N : 窓全体（幅 2N）の場を FFT。値 k の光 = 番号 2k（同じ周波数 k/N）。間の番号に落ちた光は答えにならない
  板 2N    : 板そのものも 2N の幅に広げる（比べ: 「層を大きくする」）
指標: 表との差、値に届いた光、出口で板の幅の外にある光の割合（学習の前と後）。
結果の置き場: results/exp48.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp48_big_lens.py
"""
import json, math, os, sys, time

import torch

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp21_viewpoint.py"), encoding="utf-8").read().split("\nres = dict(exp=")[0]
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)
OBJ, VAL, VIEW, TABLE = ns["OBJ"], ns["VAL"], ns["VIEW"], ns["TABLE"]
N, K, dz, steps = 128, 20, 64.0, 600; L = 2 * N
slot = {(o, v): 6 * i + VIEW[v] for i, o in enumerate(OBJ) for v in VIEW}; slot.update({x: 30 + j for j, x in enumerate(VAL)})
pairs = [(o, v) for o in OBJ for v in VIEW]
H = torch.exp(1j * dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
yy = torch.arange(N, dtype=torch.float32)
u0 = torch.stack([torch.exp(1j * 2 * math.pi * slot[p] * yy / N) / math.sqrt(N) for p in pairs]).to(torch.complex64)
vb = torch.tensor([slot[x] for x in VAL])
q = torch.zeros(len(pairs), len(VAL))
for k, (o, v) in enumerate(pairs):
    for x, p in TABLE[o][v].items():
        q[k, VAL.index(x)] = p


def run(P, lens, wide):
    u = torch.nn.functional.pad(u0, (0, L - N))
    for k in range(K):
        if wide:
            u = u * torch.exp(1j * P[:, k])
        else:
            u = torch.cat([u[:, :N] * torch.exp(1j * P[:, k]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
    outside = float((u[:, N:].abs() ** 2).sum(1).mean())
    if lens == "N":
        W = torch.fft.fft(u[:, :N], dim=-1, norm="ortho").abs()[:, vb] ** 2
    else:
        W = torch.fft.fft(u, dim=-1, norm="ortho").abs()[:, 2 * vb] ** 2
    return W, outside


out = "results/exp48.txt"
for name, lens, wide in [("レンズ N（これまで）", "N", False), ("レンズ 2N（大きなレンズ）", "2N", False), ("板 2N + レンズ 2N", "2N", True)]:
    for seed in range(3):
        t0 = time.time(); g = torch.Generator().manual_seed(seed)
        P = torch.nn.Parameter(math.pi * (2 * torch.rand(L if wide else N, K, generator=g) - 1)); opt = torch.optim.Adam([P], lr=0.01)
        with torch.no_grad():
            _, out0 = run(P, lens, wide)
        for step in range(steps):
            W, _ = run(P, lens, wide)
            p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
            loss = -(torch.cat([q, torch.zeros(len(q), 1)], 1) * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W, out1 = run(P, lens, wide)
            tv = float(((W / W.sum(1, keepdim=True)) - q).abs().sum(1).mean() / 2); light = float(W.sum(1).mean())
        r = dict(exp="48", kind=name, seed=seed, tv=tv, light=light, outside_before=out0, outside_after=out1, sec=time.time() - t0)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name} seed {seed}: 表との差 {tv:.3f}  値に届いた光 {100 * light:.1f}%  板の外の光 学習前 {100 * out0:.0f}% → 後 {100 * out1:.0f}%  {r['sec']:.0f} 秒", flush=True)
