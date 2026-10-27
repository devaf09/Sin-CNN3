"""実験10: 複素数の勾配 — torch は何を微分しているか（学習の前の確認、M0）

着想: M0 の残り。学習器では光（複素数）を通して、最後に明るさ（実数）の損失を測る。学習する数を複素数で持つと、torch の勾配が何を表すかを知っておく必要がある。
方法: 損失 L（実数）を、複素数 z の関数として 4 つ用意し、torch の z.grad と、
      (1) z を実部 x・虚部 y の 2 つの実数として持った版の勾配 ∂L/∂x + i·∂L/∂y、(2) 差分の近似（x と y を 1e-6 ずつ動かす）、(3) 手計算の式、を比べる。
        |z|²（式 2z）、|w·z|²（式 2|w|²z）、|e^(iz)|²（= e^(−2y)、式 −2i·e^(−2y)）、Σ|FFT(z)|²（norm="ortho" なら Σ|z|² と同じ → 式 2z）
      さらに、学習する数を実数の位相 θ で持つ版 L = |Σ e^(iθ_n)|²（光を位相の板に通して 1 点に集めた明るさ）の勾配を、式 −2·Im(conj(S)·e^(iθ_n)) と比べる。
指標: 差の最大値。
結果の置き場: results/exp10.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp10_complex_grad.py
"""
import argparse, json, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp10.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time(); torch.manual_seed(a.seed)
N = 8
z0 = torch.randn(N, dtype=torch.complex128); w = torch.randn(N, dtype=torch.complex128)
funcs = {"|z|²": (lambda z: (z.abs() ** 2).sum(), lambda z: 2 * z),
         "|w·z|²": (lambda z: ((w * z).abs() ** 2).sum(), lambda z: 2 * w.abs() ** 2 * z),
         "|e^(iz)|²": (lambda z: (torch.exp(1j * z).abs() ** 2).sum(), lambda z: -2j * torch.exp(-2 * z.imag)),
         "Σ|FFT(z)|²（ortho）": (lambda z: (torch.fft.fft(z, n=N, dim=-1, norm="ortho").abs() ** 2).sum(), lambda z: 2 * z)}
res = dict(exp="10", args=vars(a), rows=[])
for name, (f, formula) in funcs.items():
    z = z0.clone().requires_grad_(True); f(z).backward(); g_c = z.grad                       # torch の複素数の勾配
    x = z0.real.clone().requires_grad_(True); y = z0.imag.clone().requires_grad_(True)
    f(torch.complex(x, y)).backward(); g_r = x.grad + 1j * y.grad                              # 実部・虚部を別々の実数として
    h = 1e-6; g_d = torch.zeros(N, dtype=torch.complex128)                                    # 差分の近似
    for n in range(N):
        e = torch.zeros(N, dtype=torch.complex128); e[n] = h
        g_d[n] = (f(z0 + e) - f(z0 - e)) / (2 * h) + 1j * (f(z0 + 1j * e) - f(z0 - 1j * e)) / (2 * h)
    r = dict(f=name, vs_real=float((g_c - g_r).abs().max()), vs_diff=float((g_c - g_d).abs().max()), vs_formula=float((g_c - formula(z0)).abs().max()))
    res["rows"].append(r)
    print(f"{name:<18} torch の勾配と: 実部・虚部の版 {r['vs_real']:.1e}  差分 {r['vs_diff']:.1e}  式 {r['vs_formula']:.1e}")
# 学習する数を実数の位相で持つ: L = |S|²、S = Σ e^(iθ_n)。式: ∂L/∂θ_n = 2·Re(conj(S)·i·e^(iθ_n)) = −2·Im(conj(S)·e^(iθ_n))
th = torch.randn(N, dtype=torch.float64, requires_grad=True)
S = torch.exp(1j * th).sum(); (S.abs() ** 2).backward()
formula = -2 * (torch.conj(S) * torch.exp(1j * th)).imag
err_phase = float((th.grad - formula.detach()).abs().max())
print(f"位相 θ（実数）で持つ版: torch の勾配と式の差 {err_phase:.1e}")
res.update(phase_err=err_phase, sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
