"""実験15: 関係性概念 — 1 つの入力に 1 個以上の点を対応させる（第 1 段階）、縦の周波数を少しずらして別の点に焦点を合わせる（第 2 段階）

着想: 2026-10-27 持ち主の依頼「関係性概念とでもいうべきもの（ホログラムで、見る方向が変わると出てくる値が変わるのに似ている）」。
      第 1 段階: A → B, C, D / B → C / D → A, E, F / X → Y, Z。入力と出力は同じ仕組み（同じ位置に並べた光の点）で表し、出力をそのまま次の入力にできる形にする。
      第 2 段階: 参照光の代わりに、入力の縦の周波数を少しずらす（光を少し傾けて入れる = ホログラムを見る方向を変える）。
      ずらしたときは A → G, H（持ち主の例）、B → F、D → B, C、X → E, F, G（A 以外は私が決めた例）に焦点を合わせる。
方法: 器は実験12〜14 と同じ（縦 100 画素、100 列、列の間 --dz 画素、周波数 1〜16、縦横とも sin と cos）。曲がる処理なし。
      記号 11 個（A B C D E F G H X Y Z）を縦に約 8 画素おきに並べ、入力はその位置の光の点（幅 1.5 画素）。出力は右端の同じ位置の窓（4 画素）。
      ずらし: 入力に e^(i·2π·δ·y/100) を掛ける（100 画素で δ 周ぶん傾ける）。δ = 0 なら第 1 段階だけ。
      損失: 答えの点（何個でも等しく）の分布と、窓に入った光の割合（窓の外は「はずれ」として数える）の交差エントロピー。Adam（学習率 0.01、--steps 回）。
指標: 点の組がぴったり当たったか（明るい順に答えの数だけ選んだ組 = 答えの組）、答えの窓に入った光の割合、答えの窓の一番暗いもの ÷ ほかの窓の一番明るいもの（> 1 なら見分けられる）。
      学習前と後の、光の 2 次元マップ・出力・格子・学習の経過を results/exp15.txt に残す。
結果の置き場: results/exp15.txt。図は tools/fig_exp15.py → docs/images/exp15_*.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp15_relations.py --shifts 0           （第 1 段階）
      CUDA_VISIBLE_DEVICES= python experiments/exp15_relations.py --shifts 0.5,1,2,4    （第 2 段階、ずらしの量を変えて）
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
ap.add_argument("--shifts", default="0", help="第 2 段階のずらしの量 δ（100 画素で何周）。0 なら第 1 段階だけ")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp15.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
SYM = list("ABCDEFGHXYZ"); at = {s: int(round(8 + 84 * i / (len(SYM) - 1))) for i, s in enumerate(SYM)}   # 記号の縦の位置
win = torch.zeros(len(SYM), N)
for i, s in enumerate(SYM):
    win[i, at[s] - 2:at[s] + 2] = 1
REL0 = {"A": "BCD", "B": "C", "D": "AEF", "X": "YZ"}                                  # 第 1 段階（持ち主の例）
REL1 = {"A": "GH", "B": "F", "D": "BC", "X": "EFG"}                                  # 第 2 段階（ずらしたとき。A → G, H が持ち主の例）


def spot(s, delta=0.0):
    return (torch.exp(-((pos - at[s]) / 1.5) ** 2) * torch.exp(1j * 2 * math.pi * delta * pos / N)).to(torch.complex64)


def run(A, u0, keep_map=False):
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
    u = torch.nn.functional.pad(u0, (0, L - N)); maps = []
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
        if keep_map:
            maps.append((u[:, :N].abs() ** 2).detach())
    return u[:, :N], G, (torch.stack(maps, -1) if keep_map else None)


def measure(out, tgt):
    I = out.abs() ** 2; Wi = I @ win.T                                                 # 窓ごとの明るさ
    ok, frac, ratio = [], [], []
    for n, t in enumerate(tgt):
        idx = [SYM.index(c) for c in t]; rest = [k for k in range(len(SYM)) if k not in idx]
        ok.append(set(Wi[n].argsort(descending=True)[:len(idx)].tolist()) == set(idx))
        frac.append(float(Wi[n, idx].sum() / I[n].sum())); ratio.append(float(Wi[n, idx].min() / Wi[n, rest].max()))
    return float(np.mean(ok)), float(np.mean(frac)), float(np.mean(ratio)), Wi


res = dict(exp="15", args=vars(a), rows=[])
for delta in [float(d) for d in a.shifts.split(",")]:
    pairs = [(s, 0.0, t) for s, t in REL0.items()] + ([(s, delta, t) for s, t in REL1.items()] if delta > 0 else [])
    u0 = torch.stack([spot(s, d) for s, d, _ in pairs]); tgt = [t for _, _, t in pairs]
    q = torch.zeros(len(pairs), len(SYM) + 1)
    for n, t in enumerate(tgt):
        for c in t:
            q[n, SYM.index(c)] = 1 / len(t)                                            # 答えの点に等しく（最後の列は「窓の外」）
    for seed in range(a.seeds):
        t0 = time.time(); g = torch.Generator().manual_seed(seed)
        A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
        with torch.no_grad():
            out_b, G_b, map_b = run(A, u0, keep_map=(seed == 0)); before = measure(out_b, tgt)
        curve = []
        for step in range(a.steps):
            out, _, _ = run(A, u0); I = out.abs() ** 2; Wi = I @ win.T
            p = torch.cat([Wi, (I.sum(1) - Wi.sum(1))[:, None]], 1) / I.sum(1, keepdim=True)
            loss = -(q * torch.log(p + 1e-9)).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if step % 25 == 0 or step == a.steps - 1:
                curve.append([step, float(loss.detach()), measure(out.detach(), tgt)[0]])
        with torch.no_grad():
            out_a, G_a, map_a = run(A, u0, keep_map=(seed == 0)); after = measure(out_a, tgt)
        r = dict(delta=delta, seed=seed, pairs=[[s, d, t] for s, d, t in pairs], before=before[:3], after=after[:3],
                 win_before=before[3].tolist(), win_after=after[3].tolist(), curve=curve, sec=time.time() - t0)
        if seed == 0:
            os.makedirs("results/exp15_maps", exist_ok=True)                            # 大きい配列は別のファイルに（16 ビットで）
            np.savez_compressed(f"results/exp15_maps/delta{delta:g}_seed0.npz", G_before=G_b.numpy(), G_after=G_a.numpy(),
                                map_before=map_b.numpy().astype(np.float16), map_after=map_a.numpy().astype(np.float16),
                                in_I=(u0.abs() ** 2).numpy(), out_before=(out_b.abs() ** 2).numpy(), out_after=(out_a.abs() ** 2).numpy())
            r["maps"] = f"results/exp15_maps/delta{delta:g}_seed0.npz"
        res["rows"].append(r)
        print(f"δ={delta:<4} seed {seed}: 組がぴったり {100 * before[0]:5.1f} → {100 * after[0]:5.1f}%  答えの窓の光 {100 * before[1]:4.1f} → {100 * after[1]:4.1f}%  "
              f"見分け（暗い答え ÷ 明るいはずれ）{before[2]:.2f} → {after[2]:.2f}  {r['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
