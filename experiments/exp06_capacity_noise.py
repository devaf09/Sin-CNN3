"""実験06: 何本の数を 1 本の信号に重ねて載せ、読み出せるか — 雑音・粗い精度で、読み出しの誤差がどう増えるか

着想: 実験02 で、1 本の周波数に振幅と位相の 2 つの数を載せて読めた。実験04・05 で、違う周波数は掛けて足すと打ち消し合う（混ざらない）。
      では K 本の周波数に K 個の数（振幅と位相 = 複素数 1 個）を重ねたら、何本まで読めるか。雑音・粗い精度（アナログな装置を想定）でどう崩れるか。
方法: N = 64。数 c_k = a_k·e^(iφ_k)（a は 0.5〜1、φ はでたらめ）を周波数 k = 1〜K に載せる: x[n] = Σ a_k·cos(2π·k·n/N + φ_k)。
      読み出し: ĉ_k = (2/N)·FFT(x)[k]（norm="backward"）。雑音: x に標準偏差 σ の正規分布の雑音を足す。粗い精度: x を b ビット（2^b 段）に丸める。
      2 つの約束で比べる: 「1 本ごとの振幅がそのまま」（本数が増えると信号が大きくなる）と「信号全体の明るさを一定」（√(平均の 2 乗) = 1。光の量に上限がある装置）。
式の予想: 読み出しの誤差の大きさ（複素数の差の √平均 2 乗）= 2σ/√N。本数 K によらない（周波数どうしが混ざらないため）。
          丸めの雑音は σ = 1 段の幅 / √12 として同じ式。明るさ一定なら、1 本の振幅が 1/√K 倍に縮むので、相対誤差が √K 倍になる。
指標: 誤差（実験）と式の予想、相対誤差（誤差 / 振幅の平均）。3 seed の平均 ± 標準誤差。
結果の置き場: results/exp06.txt。図は tools/fig_exp06.py → docs/images/exp06_capacity_noise.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp06_capacity_noise.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=64)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--trials", type=int, default=200, help="1 seed あたり、でたらめな数の組を何回作るか")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp06.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
t0, N = time.time(), a.N
n = np.arange(N)
res = dict(exp="06", args=vars(a), rows=[])
for mode in ("1 本ごとの振幅", "明るさ一定"):
    for K in (1, 4, 16, 31):
        for noise in ("σ=0", "σ=0.01", "σ=0.1", "σ=0.3", "8 ビット", "4 ビット", "2 ビット"):
            errs, rels, preds = [], [], []
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                e_all, a_all, p_all = [], [], []
                for _ in range(a.trials):
                    c = rng.uniform(0.5, 1, K) * np.exp(2j * math.pi * rng.random(K))      # 載せる数 K 個
                    k = np.arange(1, K + 1)
                    x = (np.abs(c)[:, None] * np.cos(2 * math.pi * k[:, None] * n / N + np.angle(c)[:, None])).sum(0)
                    if mode == "明るさ一定":
                        s = 1 / np.sqrt((x ** 2).mean()); x, c = x * s, c * s               # 信号全体を √平均 2 乗 = 1 に
                    if noise.startswith("σ"):
                        sig = float(noise[2:]); x = x + sig * rng.standard_normal(N)
                    else:
                        b = int(noise.split()[0]); top = np.abs(x).max()
                        step = 2 * top / (2 ** b - 1)                                       # 1 段の幅（−top〜+top を 2^b 段に）
                        x = np.round((x + top) / step) * step - top; sig = step / math.sqrt(12)
                    c_hat = 2 / N * np.fft.fft(x, n=N, norm="backward")[1:K + 1]           # 読み出し（長さ N、正規化なし）
                    e_all.append(np.abs(c_hat - c) ** 2); a_all.append(np.abs(c)); p_all.append(2 * sig / math.sqrt(N))
                err = float(np.sqrt(np.concatenate(e_all).mean()))
                errs.append(err); rels.append(err / float(np.concatenate(a_all).mean())); preds.append(float(np.mean(p_all)))
            m = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))))
            r = dict(mode=mode, K=K, noise=noise, err=m(errs), rel=m(rels), pred=m(preds))
            res["rows"].append(r)
            print(f"{mode:<8} K={K:>2} {noise:<7} 誤差 {r['err'][0]:.4f}±{r['err'][1]:.4f}（式 {r['pred'][0]:.4f}）  相対 {100 * r['rel'][0]:6.2f}%")
res.update(sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
