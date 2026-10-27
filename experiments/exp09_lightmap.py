"""実験09: 周波数格子の小さな学習器の器（学習なし）— 光の 2 次元マップと、明るさが曲がる処理の有無

着想: 2026-10-27 持ち主の案。入力 = 縦 100 画素の列、処理部 = 縦 sin 周波数 × 横 cos 周波数の格子（100 × 100）、
      光は左から入って右へ 1 列ずつ進み、出力は右端の列。明るさが曲がる処理（持ち主の案）の版も作る。まず学習せずに器を作って中を見る。
方法: 格子の値 G[y, x] = Σ_ij A_ij·sin(2π·i·y/100)·cos(2π·j·x/100)（i, j = 1〜F。今回は A をでたらめに決める）。
      列 x で、光の列 u に位相の遅れ G[:, x] を掛ける（明るさは変えない板）→ 縦方向に 1 列ぶん回折（角スペクトル法、波長 1 画素、
      端で回り込まないよう縦を 2 倍にゼロを詰める）→ 曲がる処理（2 種類: 飽和 = 明るいほど通りにくい u / √(1 + |u|²/s)、
      カー効果 = 明るいほど遅れる u·e^(iγ|u|²)。カー効果は明るさを変えず、明るい所で光の進む向きが曲がる）→ 次の列。
      重ね合わせの確かめ: 入力 a・b・a + b を通し、出力(a + b) と 出力(a) + 出力(b) の差を測る（曲がる処理なしなら 0 のはず = 1 枚の表と同じ）。
指標: 光の 2 次元マップ（明るさ）、重ね合わせの差（相対）、出力の明るさの合計（板は位相だけなので、なしなら入力と同じはず）。
結果の置き場: results/exp09.txt。図は tools/fig_exp09.py → docs/images/exp09_lightmap.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp09_lightmap.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--Ny", type=int, default=100)
ap.add_argument("--Nx", type=int, default=100)
ap.add_argument("--F", type=int, default=8, help="使う周波数の数（縦・横とも 1〜F。持ち主の案の最大は 100）")
ap.add_argument("--amp", type=float, default=0.5, help="A_ij の大きさ（位相の遅れ [ラジアン] の目安）")
ap.add_argument("--sat", type=float, default=1.0, help="飽和の明るさ s（小さいほど強く曲がる）")
ap.add_argument("--gamma", type=float, default=2.0, help="カー効果の強さ γ（明るさ 1 で γ ラジアン遅れる）")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp09.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time(); torch.manual_seed(a.seed)
Ny, Nx, F = a.Ny, a.Nx, a.F
y = torch.arange(Ny, dtype=torch.float32)[:, None]; x = torch.arange(Nx, dtype=torch.float32)[None, :]
i = torch.arange(1, F + 1, dtype=torch.float32)
A = a.amp * torch.randn(F, F) / F                                                  # 学習する数（今回はでたらめに固定）
S = torch.sin(2 * math.pi * i[:, None] * y[None, :, 0] / Ny)                        # (F, Ny) 縦の sin
C = torch.cos(2 * math.pi * i[:, None] * x[0][None, :] / Nx)                        # (F, Nx) 横の cos
G = S.T @ A @ C                                                                     # (Ny, Nx) 格子の値 = Σ A_ij sin·cos
L = 2 * Ny                                                                          # 縦を 2 倍にゼロを詰める（回り込み防止、実験07）
ky = 2 * math.pi * torch.fft.fftfreq(L)                                             # 縦の空間周波数 [ラジアン/画素]
k = 2 * math.pi / 1.0                                                               # 波長 1 画素
H = torch.exp(1j * torch.sqrt((k ** 2 - ky ** 2).clamp(min=0)))                     # 1 列（距離 1 画素）進む回折。|ky| ≤ π < k なので全部進む


def run(u0, bend):
    """入力の列 u0 (Ny,) を左から右へ通す。光の 2 次元マップ (Ny, Nx) の明るさと、出力の列を返す。"""
    u = torch.zeros(L, dtype=torch.complex64); u[:Ny] = u0
    I = torch.zeros(Ny, Nx)
    for c in range(Nx):
        u[:Ny] = u[:Ny] * torch.exp(1j * G[:, c])                                   # 格子の列を通す（位相の遅れ）
        u = torch.fft.ifft(torch.fft.fft(u, n=L, norm="ortho") * H, n=L, norm="ortho")   # 1 列ぶん回折（ortho の行き帰り）
        if bend == "飽和":
            u = u / torch.sqrt(1 + u.abs() ** 2 / a.sat)                            # 明るいほど通りにくい
        elif bend == "カー効果":
            u = u * torch.exp(1j * a.gamma * u.abs() ** 2)                          # 明るいほど遅れる（明るさは変えない）
        I[:, c] = u[:Ny].abs() ** 2
    return I, u[:Ny].clone()


ua = torch.zeros(Ny, dtype=torch.complex64); ua[30] = 1                             # 入力 a: 30 番の画素に光の点
ub = torch.zeros(Ny, dtype=torch.complex64); ub[60:70] = 0.3                        # 入力 b: 60〜69 番に弱い帯
res = dict(exp="09", args=vars(a), rows=[])
for bend in ("なし", "飽和", "カー効果"):
    (Ia, oa), (Ib, ob), (Iab, oab) = run(ua, bend), run(ub, bend), run(ua + ub, bend)
    sup = float((oab - oa - ob).abs().norm() / oab.abs().norm())                    # 重ね合わせの差（相対）
    e_in, e_out = float((ua.abs() ** 2).sum()), float((oa.abs() ** 2).sum())
    r = dict(bend=bend, superpos_err=sup, energy_in_a=e_in, energy_out_a=e_out,
             I_a=Ia.tolist(), I_b=Ib.tolist(), I_ab=Iab.tolist(), out_ab=(oab.abs() ** 2).tolist(), out_sum=((oa + ob).abs() ** 2).tolist())
    res["rows"].append(r)
    print(f"曲がる処理 {bend}: 重ね合わせの差 {sup:.2e}  入力 a の明るさの合計 {e_in:.4f} → 出力 {e_out:.4f}")
res.update(G=G.tolist(), sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
