"""実験03: 割り切れない周波数の漏れと窓関数 — 長さ 16 で、等比数列の和の式と突き合わせる

着想: 学び 01 の残った疑問（2026-10-27 持ち主が OK）。N 点でちょうど整数回まわらない cos は、1 本の線にならず周りに漏れるはず。
方法: N = 16、x[n] = cos(2π·f·n/N) を f = 2（割り切れる）と f = 2.5（半端）で作り、numpy の fft（norm="backward"）に通す。
      式の予想は、cos = (右回りの針 + 左回りの針)/2 と等比数列の和 Σ r^n = (1 − r^N)/(1 − r) から（素朴な和とは別の方法）。
      窓関数: 信号の両端をなめらかに 0 に落とすハン窓 w[n] = 0.5 − 0.5·cos(2πn/N) を掛けてから fft する。
指標: 各 k の大きさ、遠くへの漏れの割合 = ±f から 3 番以上離れた番号に出たエネルギー / 全エネルギー（窓は山の近くを少し太らせるので、近くは数えない）。
結果の置き場: results/exp03.txt。図は tools/fig_exp03.py → docs/images/exp03_leak.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp03_leak.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)        # 乱数は使わない
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp03.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
t0, N = time.time(), a.N
n, k = np.arange(N), np.arange(N)


def geom(d):
    """Σ_{n=0}^{N-1} e^(2πi·d·n/N) を等比数列の和の式で（d が整数なら 0 か N）。"""
    r = np.exp(2j * math.pi * d / N)
    return np.where(np.abs(1 - r) < 1e-12, N, (1 - r ** N) / np.where(np.abs(1 - r) < 1e-12, 1, 1 - r))


res = dict(exp="03", args=vars(a), rows=[])
for f in (2.0, 2.5):
    x = np.cos(2 * math.pi * f * n / N)
    expect = 0.5 * geom(f - k) + 0.5 * geom(-f - k)                     # 式の予想（窓なし）
    dist = np.minimum(np.abs((k - f + N / 2) % N - N / 2), np.abs((k + f + N / 2) % N - N / 2))   # ±f までの番号の距離（ぐるっと回る）
    near = dist < 2.75                                                  # 3 番以上離れた所を「遠く」とする
    for win in ("なし", "ハン窓"):
        w = np.ones(N) if win == "なし" else 0.5 - 0.5 * np.cos(2 * math.pi * n / N)
        X = np.fft.fft(x * w, n=N, norm="backward")                     # 長さ N、正規化なし（明示）
        E = np.abs(X) ** 2
        r = dict(f=f, window=win, mag=np.abs(X).round(4).tolist(), leak=float(E[~near].sum() / E.sum()),
                 err_formula=float(np.abs(X - expect).max()) if win == "なし" else None)
        res["rows"].append(r)
        print(f"f = {f}  窓 {win:>3}: 大きさ = {np.abs(X).round(2).tolist()}")
        print(f"            遠くへの漏れ {100 * r['leak']:.2f}%" + (f"  式の予想との差 {r['err_formula']:.1e}" if win == "なし" else ""))
res.update(sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
