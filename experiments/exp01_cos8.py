"""実験01: 長さ 8 の cos を 1 本 — 素朴な和・numpy・torch の FFT が同じ答えを出すか

着想: M0 の最初の一歩（2026-10-27 持ち主の例）。すべての確認の土台として、いちばん簡単な信号で 3 つの計算を突き合わせる。
方法: x[n] = cos(2π·1·n/8)（8 点で 1 周する cos）を、自作の素朴な和（src/naive_dft.py）・numpy.fft.fft・torch.fft.fft に通す。
      式の予想: X[1] = X[7] = 4（= N/2）、ほかは 0。
指標: 各 k の X[k]（実部・虚部）、式の予想との差の最大値、3 つの計算どうしの差の最大値。
結果の置き場: results/exp01.txt（1 回 = 1 行の JSON）。図は tools/fig_exp01.py → docs/images/exp01_cos8.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp01_cos8.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naive_dft import naive_dft  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--k0", type=int, default=1, help="入れる cos の周波数（N 点で何周するか）")
ap.add_argument("--seed", type=int, default=0)        # 乱数は使わないが、形をそろえるため
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp01.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")                                # Windows の画面でも · や π を出せるように
t0 = time.time()

N, k0 = a.N, a.k0
x = np.array([math.cos(2 * math.pi * k0 * n / N) for n in range(N)])   # float64

X_naive = np.array(naive_dft(list(x)))                                  # 自作の素朴な和
X_np = np.fft.fft(x, n=N, norm="backward")                              # numpy: 長さ n=N、正規化なし（既定と同じだが明示）
X_th = torch.fft.fft(torch.tensor(x, dtype=torch.float64, device=a.device),
                     n=N, dim=-1, norm="backward").cpu().numpy()        # torch: 同じ約束。float64 で計算
X_exp = np.zeros(N, complex); X_exp[k0] += N / 2; X_exp[(N - k0) % N] += N / 2   # 式の予想

err = lambda A, B: float(np.abs(A - B).max())
print(f"x[n] = cos(2π·{k0}·n/{N}) = {np.round(x, 4).tolist()}")
print(f"{'k':>2} | {'式の予想':>8} | {'素朴な和':>22} | {'numpy':>22} | {'torch':>22}")
for k in range(N):
    f = lambda z: f"{z.real:+.4f} {z.imag:+.4f}i"
    print(f"{k:>2} | {X_exp[k].real:>8.1f} | {f(X_naive[k]):>22} | {f(X_np[k]):>22} | {f(X_th[k]):>22}")
res = dict(exp="01", args=vars(a), x=x.tolist(),
           X_naive=[[z.real, z.imag] for z in X_naive], X_numpy=[[z.real, z.imag] for z in X_np],
           X_torch=[[z.real, z.imag] for z in X_th],
           err_naive_vs_expect=err(X_naive, X_exp), err_numpy_vs_naive=err(X_np, X_naive),
           err_torch_vs_naive=err(X_th, X_naive), err_numpy_vs_torch=err(X_np, X_th),
           sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
print({k: f"{v:.1e}" for k, v in res.items() if k.startswith("err")})
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
