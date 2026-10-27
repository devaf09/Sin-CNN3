"""実験07: ぐるっと回る畳み込み・相関と、ゼロ詰め — 前のプロジェクトの記憶の読み出しの式も確かめる

着想: M0 の残り。実験05 で「波の側の掛け算 = 周波数の側のぐるっと回る畳み込み」を見た。逆も同じで、周波数の側で掛けると、
      波の側ではぐるっと回る畳み込み（端からはみ出した分が反対側に回り込む）になる。前のプロジェクトの回折・4f 系・記憶は、ゼロを詰めずにこれを使っていた。
方法: (A) N = 8 の x = [1..8] と h = [1, 1, 1, 0…]: FFT の掛け算、素朴なぐるっと回る和、numpy.convolve（回り込まない普通の畳み込み）、
          長さ 16 にゼロを詰めた FFT の掛け算、を並べる。
      (B) 相関: 鍵 k（±1 のでたらめ）と、鍵を 3 つずらしたもの。ifft(conj(FFT(k))·FFT(y)) が素朴な相関 Σ k[n]·y[n + s] と一致し、s = 3 に山が出るか。
      (C) 前のプロジェクトの記憶の式（SN2-Parts-2D exp87 の make_grain・recall と同じ形）を 8 × 8 で: 板 H = conj(FFT2(鍵))·FFT2(点)、
          読み出し |ifft2(FFT2(鍵)·H)|²。これが「鍵の自己相関（ぐるっと回る）を点の位置へずらしたもの」と一致するか、ゼロを詰めた版（回り込まない）とどれだけ違うか。
指標: 各計算どうしの差の最大値、回り込みの量。
結果の置き場: results/exp07.txt。図は tools/fig_exp07.py → docs/images/exp07_circular.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp07_circular.py
"""
import argparse, json, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp07.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
t0, N, rng = time.time(), a.N, np.random.default_rng(a.seed)
fft, ifft = (lambda v, m: np.fft.fft(v, n=m, norm="backward")), (lambda V, m: np.fft.ifft(V, n=m, norm="backward"))

# (A) 畳み込み
x = np.arange(1, N + 1, dtype=float); h = np.zeros(N); h[:3] = 1
circ_fft = ifft(fft(x, N) * fft(h, N), N).real                                   # FFT の掛け算（長さ N のまま）
circ_sum = np.array([sum(x[m] * h[(s - m) % N] for m in range(N)) for s in range(N)])   # ぐるっと回る素朴な和
lin = np.convolve(x, h[:3])                                                       # 普通の畳み込み（長さ N + 2）
pad_fft = ifft(fft(x, 2 * N) * fft(h, 2 * N), 2 * N).real[:N + 2]                # n=2N で後ろにゼロを詰めてから FFT
print("(A) ぐるっと回る（FFT）  ", circ_fft.round(3).tolist())
print("    ぐるっと回る（素朴な和）", circ_sum.round(3).tolist())
print("    普通の畳み込み        ", lin.round(3).tolist())
print("    ゼロを詰めた FFT       ", pad_fft.round(3).tolist())

# (B) 相関
k = rng.choice([-1.0, 1.0], N); y = np.roll(k, 3)                                # y[n] = k[n − 3]
corr_fft = ifft(np.conj(fft(k, N)) * fft(y, N), N).real
corr_sum = np.array([sum(k[n] * y[(n + s) % N] for n in range(N)) for s in range(N)])
print("(B) 相関（FFT）", corr_fft.round(3).tolist(), " 素朴な和", corr_sum.round(3).tolist())

# (C) 前のプロジェクトの記憶の式（8 × 8、鍵 1 つ・値 = 点 1 つ）
key = np.exp(2j * np.pi * rng.random((N, N))); pt = np.zeros((N, N)); pt[2, 5] = 1
H = np.conj(np.fft.fft2(key, norm="backward")) * np.fft.fft2(pt, norm="backward")  # 書き込み（exp87 と同じ形）
read = np.fft.ifft2(np.fft.fft2(key, norm="backward") * H, norm="backward")      # 読み出し（明るさは |read|²）
ac = np.array([[(np.conj(key) * np.roll(key, (-p, -q), (0, 1))).sum() for q in range(N)] for p in range(N)])  # 自己相関（ぐるっと回る）
ac_at_pt = np.roll(ac, (2, 5), (0, 1))                                            # 点の位置 (2, 5) へずらす
M = 2 * N                                                                          # ゼロを詰めた版（回り込まない）
Hp = np.conj(np.fft.fft2(key, (M, M))) * np.fft.fft2(pt, (M, M))
readp = np.fft.ifft2(np.fft.fft2(key, (M, M)) * Hp)
I, Ip = np.abs(read) ** 2, np.abs(readp) ** 2
print(f"(C) 読み出し = 自己相関を点へずらしたもの: 差 {np.abs(read - ac_at_pt).max():.1e}")
print(f"    山の高さ（点の位置）: 回る {I[2, 5]:.1f}、ゼロ詰め {Ip[2, 5]:.1f}。山以外の明るさの平均: 回る {np.delete(I.ravel(), 2 * N + 5).mean():.2f}、"
      f"ゼロ詰め {(Ip.sum() - Ip[2, 5]) / (M * M - 1):.2f}（ゼロ詰めは {M}×{M} に広がる）")
res = dict(exp="07", args=vars(a), A=dict(circ=circ_fft.tolist(), circ_sum=circ_sum.tolist(), linear=lin.tolist(), padded=pad_fft.tolist(),
           err_circ=float(np.abs(circ_fft - circ_sum).max()), err_pad=float(np.abs(pad_fft - lin).max())),
           B=dict(k=k.tolist(), corr=corr_fft.tolist(), err=float(np.abs(corr_fft - corr_sum).max())),
           C=dict(err_autocorr=float(np.abs(read - ac_at_pt).max()), I=I.tolist(), Ip=Ip.tolist()),
           sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
