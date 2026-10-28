"""実験47: 取り出しの非線形（べき p）とループで、記憶の容量は増えるか（実験08 の続き）

着想: 2026-10-28 持ち主の案。1. √ の代わりに 2 乗などで取り出す 2. 記憶ループ（板 → 回折 → 非線形 → 板へ戻す）3. 実験22 を何回も通す。
方法: 長さ N = 256、鍵 = 場の側で位相がでたらめな複素数（長さ 1）。M 組の (鍵, 値)、値の候補 V = 16。
  1 段      : 1 枚の板（4f のフィルタ）に Σ 鍵 → 値の点 を重ね書き（実験08 の「前型」）。出口の値の点の明るさで答える。
  2 段・重ね : 1 枚目の板に Σ 鍵 → 鍵ごとの点 i を重ね書き（相関面に M 個の点）。明るさ I を測り、I^p にして出し直し、
               2 枚目で点 i → 値へ（2 枚目は混線なしの理想として、値ごとに I^p を足す）。p = 1, 2, 4, ∞（いちばん明るい点だけ）。
  2 段・別々 : 鍵を 1 枚に重ねず、鍵ごとに別の記憶（体積ホログラム・多重の板のつもり。記憶の大きさ M × N）。I_i = |<鍵_i, 問い>|²。
  ループ    : 自己連想（値 = 鍵そのもの）。問い = 鍵の位相に雑音（重なり 0.5 前後）。相関の明るさ I^p で鍵を重ねて組み直し、問いに戻す。
              T 回まわして、最後にいちばん明るい鍵が正しいか。
式の予想: 重ねた板の相関面の雑音は、どの点でも分散 M/N（実験08 と同じ）。別々の記憶では、ほかの鍵との重なりの明るさ |<k_i,k_j>|² は
          平均 1/N の指数分布。べき p をかけた外れの明るさの分散は ((2p)! − (p!)²)/N^{2p}。値 1 つに集まる外れは M/V 個の和で、
          正しい値の点数 1 との差がこの和のゆれ（標準偏差 √(M/V)·√分散）に埋もれる所で崩れる。
          → 崩れ始める M ≈ V·N^{2p} / ((2p)! − (p!)²) / (2 ln V) の桁（p を 1 上げるごとに N² 倍）。N = 16 なら p = 1 で 約 740、p = 2 で 約 9500。
指標: 正解率（3 seed、問い 256 個）。ループでは、組み直した問いと正しい鍵の重なり |<k, q>|²（1 = 完全にきれい）も。結果の置き場: results/exp47.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp47_readout_nonlinear.py（N = 256）
      python experiments/exp47_readout_nonlinear.py --N 16 --Ms 8,16,64,256,1024,4096,16384,65536（別々の記憶の容量を見る）
"""
import argparse, json, math, sys, time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--V", type=int, default=16)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--Q", type=int, default=256)
ap.add_argument("--Ms", default="8,16,32,64,128,256,512,1024,2048,4096,8192,16384")
ap.add_argument("--out", default="results/exp47.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
N, V, Q = a.N, a.V, a.Q
fft, ifft = np.fft.fft, np.fft.ifft
keys = lambda M, r: np.exp(2j * math.pi * r.random((M, N))) / math.sqrt(N)


def powsel(I, p):
    """明るさ I（… × M）に非線形をかける。p = inf はいちばん明るい点だけ残す。"""
    if p == math.inf:
        return (I == I.max(-1, keepdims=True)).astype(float)
    return (I / I.max(-1, keepdims=True)) ** p                                  # 大きさをそろえてから（あふれ防止）


def corr_superposed(K, q):
    """1 枚の板に Σ_i conj(FFT k_i)·(点 i へのずらし) を重ね書き → 相関面の場。点 i の場 = Σ_j (k_j ⋆ q)(i − j)。"""
    M = len(K)
    H = (np.conj(fft(K, axis=-1)) * np.exp(-2j * math.pi * np.outer(np.arange(M), np.arange(N)) / N)).sum(0)
    return ifft(fft(q, axis=-1) * H, axis=-1)[:, :M]                            # 点 0..M−1 だけ読む（M ≤ N）


def corr_separate(K, q):
    return q @ np.conj(K).T                                                      # 鍵ごとの内積 <k_i, q>


res = []
for seed in range(a.seeds):
    r = np.random.default_rng(seed)
    for M in [int(x) for x in a.Ms.split(",")]:
        K = keys(M, r); val = r.integers(0, V, M); idx = r.integers(0, M, Q); q = K[idx]
        row = dict(exp="47", seed=seed, M=M, N=N, V=V)
        if M <= N:
            # 1 段: 鍵 → 値の点（値 c の点 = 位置 c）。読む = 値の点の明るさ
            H = (np.conj(fft(K, axis=-1)) * np.exp(-2j * math.pi * np.outer(val, np.arange(N)) / N)).sum(0)
            out = np.abs(ifft(fft(q, axis=-1) * H, axis=-1)[:, :V]) ** 2
            row["1段"] = float((out.argmax(1) == val[idx]).mean())
            I = np.abs(corr_superposed(K, q)) ** 2
            for p in [1, 2, 4, math.inf]:
                S = np.zeros((Q, V)); np.add.at(S.T, val, powsel(I, p).T)
                row[f"2段・重ね p={p}"] = float((S.argmax(1) == val[idx]).mean())
        I = np.abs(corr_separate(K, q)) ** 2
        for p in [1, 2, 3, 4]:
            S = np.zeros((Q, V)); np.add.at(S.T, val, powsel(I, p).T)
            row[f"2段・別々 p={p}"] = float((S.argmax(1) == val[idx]).mean())
        # ループ（自己連想）: 雑音の入った鍵から始めて T 回組み直す
        q0 = K[idx] * np.exp(1j * r.normal(0, 1.18, (Q, N)))                    # 位相の雑音 σ = 1.18 → 重なり e^{−σ²/2} ≈ 0.5
        for kind, corr in [("重ね", corr_superposed), ("別々", corr_separate)]:
            if kind == "重ね" and M > N:
                continue
            for p in [1, 2, 4, math.inf]:
                qq = q0.copy()
                for T in range(1, 6):
                    I = np.abs(corr(K, qq)) ** 2
                    if T in (1, 2, 5):
                        row[f"ループ・{kind} p={p} T={T}"] = float((I.argmax(1) == idx).mean())
                        row[f"重なり・{kind} p={p} T={T}"] = float((np.abs((qq * np.conj(K[idx])).sum(1)) ** 2).mean())   # 問い（組み直した鍵）と正しい鍵の重なり
                    W = powsel(I, p); qq = W @ K; qq /= np.linalg.norm(qq, axis=1, keepdims=True)   # 2 枚目: 点 i → 鍵 i で組み直す
        res.append(row)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(seed, M, {k: round(v, 2) for k, v in row.items() if isinstance(v, float)}, flush=True)
