"""実験08: 鍵と値を掛け算で結んで重ね書きする記憶 — 組の数を増やしたときの崩れ方（容量の線）を、式と実験で突き合わせる

着想: 実験06 で、長さ N の信号には N 個の数が混ざらずに入ると分かった。前のプロジェクトの記憶（実験07 で、読み出し = 鍵の自己相関と確認）は
      それよりずっと手前で崩れた。「結んで重ねる」と何が起きるのかを、HRR（Plate の縮約表現）の考え方で式にして確かめる。
方法: 長さ N = 256。M 組の (鍵 k_i, 値 v_i) を、記憶 m = Σ_i k_i ⊛ v_i（ぐるっと回る畳み込み = FFT の掛け算）に重ね書き。
      読み出し: v̂ = k_j ⋆ m（相関 = ifft(conj(FFT(k_j))·FFT(m))）。値の候補 V = 256 個の中で一番近いものを答える。5 つの作り方:
        HRR       : 鍵・値とも、でたらめな実数（正規分布、分散 1/N）。候補は 256 個のでたらめな値。近さ = 内積
        FHRR      : 鍵・値とも、周波数の側の大きさがすべて 1 の実数（位相だけでたらめ）。候補・近さは同じ
        前型      : 前のプロジェクトの形。鍵 = 場の側で位相がでたらめ（大きさ一定）の複素数、値 = 256 か所のどれかに光の点。近さ = 明るさ |v̂|²
        前型+鍵 1 : 値は点のまま、鍵だけ周波数の側の大きさを 1 にそろえる（学び 00 の気になる所: 鍵の作り方）
        前型+ゼロ : 前型を長さ 2N にゼロを詰めて計算（回り込まない装置のつもり。板は 2 倍の大きさ）
式の予想: 読み出しの点数は「正しい値 = 1 + 雑音」「ほかの候補 = 雑音」。雑音の分散を組の数 M から式で出し（下の pred_var）、
          正規分布として「正しい点数がほかの 255 個すべてより大きい確率」を計算する。
比較相手: デジタルの辞書（鍵をそのまま覚えて一番近い鍵を探す: 正確な鍵なら 100%）、偶然 1/256 = 0.4%。
指標: 正解率（3 seed の平均 ± 標準誤差）、雑音の分散（実験と式）。
結果の置き場: results/exp08.txt。図は tools/fig_exp08.py → docs/images/exp08_memory_capacity.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp08_memory_capacity.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp08.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
t0, N, V = time.time(), a.N, a.N
fft, ifft = (lambda x, n=None: np.fft.fft(x, n=n, axis=-1, norm="backward")), (lambda X: np.fft.ifft(X, axis=-1, norm="backward"))


def vecs(kind, n, rng):
    """でたらめな鍵・値を n 本。"""
    if kind == "gauss":   return rng.standard_normal((n, N)) / math.sqrt(N)                       # 分散 1/N（長さ ≈ 1）
    if kind == "flat":    g = fft(rng.standard_normal((n, N))); return ifft(g / np.abs(g)).real     # 周波数の大きさ 1、実数
    if kind == "phase":   return np.exp(2j * math.pi * rng.random((n, N))) / math.sqrt(N)          # 場の側で大きさ一定
    if kind == "flatc":   return ifft(np.exp(2j * math.pi * rng.random((n, N))))                   # 周波数の大きさ 1、複素数


def pred_var(kind, M):
    """式の予想: (正しい候補の雑音の分散, ほかの候補の雑音の分散)。導き方は docs/lessons/08。"""
    return {"HRR": ((M + 5) / N, (M + 1) / N), "FHRR": ((M - 1) / N, M / N),
            "前型": ((M - 1) / N, M / N), "前型+鍵1": ((M - 1) / N, (M - 1) / N),
            "前型+ゼロ": ((M - 1) / (2 * N), M / (2 * N))}[kind]


def pred_acc(kind, M, rng, S=20000):
    """正規分布の模型で、正しい点数がほかの V−1 個より大きい確率（正しい点数をサンプルして平均）。"""
    vc, vw = pred_var(kind, M)
    if kind in ("HRR", "FHRR"):                                                  # 実数の内積: P(ほか < x) = Φ(x/σ)
        x = torch.tensor(1 + math.sqrt(vc) * rng.standard_normal(S))
        return float((torch.special.ndtr(x / math.sqrt(vw)) ** (V - 1)).mean())
    z = 1 + math.sqrt(vc / 2) * (rng.standard_normal(S) + 1j * rng.standard_normal(S))   # 明るさ: P(|雑音|² < x) = 1 − e^(−x/σ²)
    return float(((1 - np.exp(-np.abs(z) ** 2 / max(vw, 1e-12))) ** (V - 1)).mean())      # vw = 0（雑音なし）なら必ず当たる


res = dict(exp="08", args=vars(a), rows=[])
for kind in ("HRR", "FHRR", "前型", "前型+鍵1", "前型+ゼロ"):
    for M in (1, 4, 16, 32, 64, 128, 256, 512):
        accs, vcs, vws = [], [], []
        for seed in range(a.seeds):
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, V, M)                                          # 各組の値は候補の何番か
            L = 2 * N if kind == "前型+ゼロ" else N
            if kind in ("HRR", "FHRR"):
                cand = vecs("gauss" if kind == "HRR" else "flat", V, rng); k = vecs("gauss" if kind == "HRR" else "flat", M, rng); v = cand[idx]
            else:
                cand = np.eye(L)[:V]; k = vecs("flatc" if kind == "前型+鍵1" else "phase", M, rng); v = cand[idx]
            K = fft(k, L)
            mem = (K * fft(v, L)).sum(0)                                         # 書き込み: Σ FFT(鍵)·FFT(値)（L 点で掛ける = 回る畳み込み）
            vh = ifft(np.conj(K) * mem)                                          # 読み出し: 相関
            score = (vh.real @ cand.T) if kind in ("HRR", "FHRR") else np.abs(vh[:, :V]) ** 2
            accs.append(float((score.argmax(1) == idx).mean()))
            if kind in ("HRR", "FHRR"):
                s_ok = score[np.arange(M), idx]; wrong = np.delete(score, idx + V * np.arange(M)).ravel()
                vcs.append(float(np.var(s_ok - 1) if M > 1 else np.mean((s_ok - 1) ** 2))); vws.append(float(np.var(wrong)))
            else:
                n_ok = vh[np.arange(M), idx] - 1; mask = np.ones((M, V), bool); mask[np.arange(M), idx] = False
                vcs.append(float(np.mean(np.abs(n_ok) ** 2))); vws.append(float(np.mean(np.abs(vh[:, :V][mask]) ** 2)))
        mse = lambda x: (float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x))))
        r = dict(kind=kind, M=M, acc=mse(accs), pred=pred_acc(kind, M, np.random.default_rng(99)),
                 var_ok=mse(vcs), var_wrong=mse(vws), pred_var=pred_var(kind, M))
        res["rows"].append(r)
        print(f"{kind:<6} M={M:>3}  正解率 {100 * r['acc'][0]:5.1f}±{100 * r['acc'][1]:.1f}%（式 {100 * r['pred']:5.1f}%）  "
              f"雑音の分散×N: 正しい {N * r['var_ok'][0]:6.1f}（式 {N * r['pred_var'][0]:6.1f}） ほか {N * r['var_wrong'][0]:6.1f}（式 {N * r['pred_var'][1]:6.1f}）")
res.update(sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
