"""実験04: 周波数の格子マップに入力の波を当てた光影 — 入力の周波数で光影がどう変わるか

着想: 2026-10-27 持ち主の案「縦横それぞれ 4 軸の sin 周波数、又は cos 周波数の格子状のマップに、入力周波数 fIn1 を投影してできる光影と、
      fIn2 を投影してできる光影の違いを比較したい」。情報がなぜ読める（処理できる）のかの入り口。
方法: 板 = 4 × 4 の区画（1 区画 16 × 16 画素）。行 i・列 j の区画には、横の周波数 fx = j+1・縦の周波数 fy = i+1 の斜めの縞
      cos(2π(fx·x + fy·y)/16)（cos の板）または sin(…)（sin の板）を描く。周波数は「16 画素で何周するか」。
      入力 = 16 × 16 の縞 cos(2π(a·x + b·y)/16 + φ)。同じ入力を全部の区画に当て（掛け算 = 透かし絵に光を通す）、区画ごとに通った光を合計する。
      合計は正にも負にもなる（負 = 影）。実際の光は負にならないが、板と光に一様な明るさを足すと「一定の明るさ + この合計の 1/4」になるだけ。
式の予想: 針の足し算（等比数列の和）で計算した値（素朴な画素の和とは別の方法）。整数の周波数なら、同じ周波数の区画だけ ±128（= 16²/2）、ほかは 0。
指標: 4 × 4 の光影の表（cos の板・sin の板）、式の予想との差の最大値。
結果の置き場: results/exp04.txt。図は tools/fig_exp04.py → docs/images/exp04_board.png・exp04_shadow.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp04_grid_shadow.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=16, help="1 区画の画素数（縦横）")
ap.add_argument("--seed", type=int, default=0)        # 乱数は使わない
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp04.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
t0, N = time.time(), a.N
y, x = np.mgrid[0:N, 0:N]                                   # 区画の中の画素の位置（縦 y、横 x）
F = [1, 2, 3, 4]                                            # 格子の周波数（横 4 軸・縦 4 軸）
inputs = {"fIn1 (a=1, b=2)": (1, 2, 0.0), "fIn2 (a=3, b=1)": (3, 1, 0.0),
          "fIn1 を 90° ずらす (a=1, b=2, φ=90°)": (1, 2, 90.0), "半端な周波数 (a=2.5, b=2)": (2.5, 2, 0.0),
          "格子にない周波数 (a=6, b=2)": (6, 2, 0.0)}


def g(d):
    """Σ_{n=0}^{N-1} e^(2πi·d·n/N)（等比数列の和）。d が整数なら N（d が N の倍数）か 0。"""
    q = np.exp(2j * math.pi * d / N)
    return N if abs(1 - q) < 1e-12 else (1 - q ** N) / (1 - q)


res = dict(exp="04", args=vars(a), F=F, rows=[])
for name, (fa, fb, ph) in inputs.items():
    phi = math.radians(ph)
    u = np.cos(2 * math.pi * (fa * x + fb * y) / N + phi)  # 入力の縞
    S = {"cos": np.zeros((4, 4)), "sin": np.zeros((4, 4))}
    E = {"cos": np.zeros((4, 4)), "sin": np.zeros((4, 4))}
    for i, fy in enumerate(F):
        for j, fx in enumerate(F):
            B = 2 * math.pi * (fx * x + fy * y) / N
            S["cos"][i, j] = (u * np.cos(B)).sum()          # 光を通して合計（素朴な画素の和）
            S["sin"][i, j] = (u * np.sin(B)).sum()
            m = np.exp(1j * phi) * g(fa - fx) * g(fb - fy)  # 差の針 Σ e^(i(A−B))
            p = np.exp(1j * phi) * g(fa + fx) * g(fb + fy)  # 和の針 Σ e^(i(A+B))
            E["cos"][i, j] = 0.5 * (m + p).real             # cosA·cosB = [cos(A−B) + cos(A+B)]/2
            E["sin"][i, j] = 0.5 * (p - m).imag             # cosA·sinB = [sin(A+B) − sin(A−B)]/2
    r = dict(input=name, a=fa, b=fb, phase_deg=ph, cos=S["cos"].round(3).tolist(), sin=S["sin"].round(3).tolist(),
             err_cos=float(np.abs(S["cos"] - E["cos"]).max()), err_sin=float(np.abs(S["sin"] - E["sin"]).max()))
    res["rows"].append(r)
    print(f"\n== {name}   式の予想との差: cos の板 {r['err_cos']:.1e}、sin の板 {r['err_sin']:.1e}")
    for k in ("cos", "sin"):
        print(f"  {k} の板（行 = 縦の周波数 1〜4、列 = 横の周波数 1〜4）")
        for i in range(4):
            print("    " + " ".join(f"{v:+8.1f}" for v in S[k][i]))
res.update(sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
