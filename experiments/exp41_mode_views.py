"""実験41: 観点 f1 × 観点 f2（2D の観点周波数）— f2 を AI の状態（モード）にして、同じ物・同じ観点でも答え方を変えられるか

着想: 2026-10-28 持ち主の案「観点周波数 f1 × 観点周波数 f2 は 2D。f2 を AI 本体の現状に合わせたパラメータとして使いたい（喜怒哀楽・焦り・最短モード・睡眠対応など）。
      エッジデバイスや状況によっては機能を制限したほうが良いので f2 にまとめる」。
      光の波どうしを掛けると周波数は足し算になる（実験05）ので、f1 × f2 = 「f1 + f2 だけずらす」。f1 = 1〜5、f2 = 0, 6, 12, …（6 おき）にすると、
      どの組も重ならない（桁を分ける置き方）。f2 = 0 を「ふつう」にすれば、エッジでは f2 を使わないだけで機能を制限できる。
方法: 実験21 の表（物 5・観点 f1 5・値 12）。モード f2 を M 個（2・4・8）:
        0 ふつう（表の割合）、1 最短（割合を 4 乗して鋭く）、2 考え中（割合をならす: 0.4 乗）、3 睡眠（光の 90% を出口の外へ = 反応を弱く）、
        4〜7 温度 0.5・2・3・0.25（割合の 1/温度 乗）
      入口 = 物の波を f1 + 6·f2 だけずらす。記号は詰めて置く（実験33）: 物 o の区画 = 6M 個、その中に f1 + 6·f2、値はその後ろ。縦 N = 30M + 16 以上（最小 100）。
      板は画素ごとの板（実験28 で規模に強かった形）、列 20、列の間 64 画素。Adam（0.01）1500 回、3 seed。
指標: モードごとの、目標の分布（値 12 + 出口の外）との差（全変動距離）、睡眠モードで値に届いた光。
結果の置き場: results/exp41.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp41_mode_views.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--Ms", default="2,4,8")
ap.add_argument("--cols", type=int, default=20)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp41.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp21_viewpoint.py"), encoding="utf-8").read().split("\nres = dict(exp=")[0]
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)
OBJ, VAL, VIEW, TABLE = ns["OBJ"], ns["VAL"], ns["VIEW"], ns["TABLE"]
MODES = ["ふつう", "最短", "考え中", "睡眠", "温度 0.5", "温度 2", "温度 3", "温度 0.25"]


def mode_target(p, m):
    """表の割合 p（値 12 個）から、モード m の目標（値 12 個 + 出口の外 1 個）を作る。"""
    p = np.asarray(p, dtype=float); sup = p > 0
    if MODES[m] == "睡眠":
        return np.append(0.1 * p, 0.9)
    pw = {"ふつう": 1, "最短": 4, "考え中": 0.4, "温度 0.5": 2, "温度 2": 0.5, "温度 3": 1 / 3, "温度 0.25": 4}[MODES[m]]
    if MODES[m] == "考え中":
        q = np.where(sup, p ** pw, 0) + 0.02 * np.ones_like(p)                          # 候補を広く残す（ほかの値にも少し）
    else:
        q = np.where(sup, p ** pw, 0)
    return np.append(q / q.sum(), 0.0)


res = dict(exp="41", args=vars(a), rows=[])
for M in [int(m) for m in a.Ms.split(",")]:
    N = max(100, 30 * M + 16); L = 2 * N
    pos = torch.arange(N, dtype=torch.float32)
    H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
    slot = lambda o, v, m: OBJ.index(o) * 6 * M + 6 * m + VIEW[v]                           # 物の区画の中に f1 + 6·f2
    vfreq = [5 * 6 * M + j for j in range(len(VAL))]                                      # 値は物の区画の後ろ
    wave = lambda k: (torch.exp(1j * 2 * math.pi * k * pos / N) / math.sqrt(N)).to(torch.complex64)
    keys = [(o, v, m) for o in OBJ for v in VIEW for m in range(M)]
    u0 = torch.stack([wave(slot(o, v, m)) for o, v, m in keys])
    tgt = torch.tensor(np.stack([mode_target([TABLE[o][v].get(x, 0) for x in VAL], m) for o, v, m in keys]), dtype=torch.float32)
    vb = torch.tensor(vfreq)

    def run(P, u):
        u = torch.nn.functional.pad(u, (0, L - N))
        for c in range(a.cols):
            u = torch.cat([u[:, :N] * torch.exp(1j * P[:, c]), u[:, N:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
        W = torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs()[:, vb] ** 2
        return torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)                # 値 12 + 出口の外

    for seed in range(a.seeds):
        t0 = time.time(); g = torch.Generator().manual_seed(seed)
        P = torch.nn.Parameter(0.1 * torch.randn(N, a.cols, generator=g)); opt = torch.optim.Adam([P], lr=0.01)
        for step in range(a.steps):
            loss = -(tgt * torch.log(run(P, u0) + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            Pn = run(P, u0).numpy(); T = tgt.numpy()
            per = {}
            for m in range(M):
                idx = [i for i, k in enumerate(keys) if k[2] == m]
                if MODES[m] == "睡眠":
                    per[MODES[m]] = dict(light=float(1 - Pn[idx, -1].mean()), tv=float(np.abs(Pn[idx] - T[idx]).sum(1).mean() / 2))
                else:
                    D = Pn[idx, :-1] / Pn[idx, :-1].sum(1, keepdims=True); Q = T[idx, :-1]
                    per[MODES[m]] = dict(tv=float(np.abs(D - Q).sum(1).mean() / 2), light=float(1 - Pn[idx, -1].mean()),
                                         top=float(np.mean(D.argmax(1) == Q.argmax(1))))
        r = dict(M=M, N=N, seed=seed, per_mode=per, sec=time.time() - t0); res["rows"].append(r)
        print(f"M={M}（縦 {N}）seed {seed}: " + "  ".join(f"{k} 差{v['tv']:.2f}・光{100 * v['light']:.0f}%" for k, v in per.items()) + f"  {r['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
