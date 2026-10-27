"""実験05: 波と波の掛け算で、周波数がずれる（差と和）— 周波数の側では「ぐるっと回る畳み込み」になる

着想: 実験04 で、格子の縞と入力を掛けると「差の周波数」と「和の周波数」ができ、差が 0 の区画だけが光った。
      持ち主の学習器の案（入力 → 逆フーリエ変換 →（周波数格子 + 何か）× n → フーリエ変換）の「格子 × 入力」は、
      この掛け算で周波数をずらして混ぜる働きをするはず。1 次元の小さな例で、ずれ方・位相・回り込み・混ざり方を式と突き合わせる。
方法: N = 32。入力 x と格子の波 g を画素ごとに掛け、FFT（norm="backward"）。
      式の予想 1: cos(2π·a·n/N + φ1)·cos(2π·c·n/N + φ2) = [cos(差 + (φ1 − φ2)) + cos(和 + (φ1 + φ2))] / 2 → 差と和の番号に N/4 ずつ。
      式の予想 2（別の方法）: FFT(x·g) = (1/N)·（X と G のぐるっと回る畳み込み）。畳み込みは素朴な和で計算する。
指標: 山の番号・大きさ・位相、予想 2 との差の最大値。
結果の置き場: results/exp05.txt。図は tools/fig_exp05.py → docs/images/exp05_mix.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp05_mix.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)        # 乱数は使わない
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp05.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
t0, N = time.time(), a.N
n = np.arange(N)
w = lambda f, deg=0.0: np.cos(2 * math.pi * f * n / N + math.radians(deg))
cases = {"5 × 3": (w(5), w(3)),
         "5 (60°) × 3 (20°)": (w(5, 60), w(3, 20)),
         "12 × 10（和 22 が 16 を超える）": (w(12), w(10)),
         "(3 + 7) × 2（2 本の入力を混ぜる）": (w(3) + w(7), w(2))}


def circ_conv(X, G):
    """ぐるっと回る畳み込み C[k] = Σ_m X[m]·G[(k − m) mod N]（素朴な和）。"""
    return np.array([sum(X[m] * G[(k - m) % N] for m in range(N)) for k in range(N)])


res = dict(exp="05", args=vars(a), rows=[])
for name, (x, g) in cases.items():
    Y = np.fft.fft(x * g, n=N, norm="backward")                  # 掛けてから FFT（長さ N、正規化なし）
    X, G = np.fft.fft(x, n=N, norm="backward"), np.fft.fft(g, n=N, norm="backward")
    err = float(np.abs(Y - circ_conv(X, G) / N).max())           # 予想 2 との差
    peaks = [(k, round(float(abs(z)), 3), round(math.degrees(math.atan2(z.imag, z.real)), 1))
             for k, z in enumerate(Y) if abs(z) > 1e-9 and k <= N // 2]   # 正の周波数の側だけ表示
    res["rows"].append(dict(case=name, x=x.tolist(), g=g.tolist(), Y=[[z.real, z.imag] for z in Y], peaks=peaks, err_conv=err))
    print(f"{name:<30} 山 (番号, 大きさ, 位相[度]) = {peaks}   畳み込みの式との差 {err:.1e}")
res.update(sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
