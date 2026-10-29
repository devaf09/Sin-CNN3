"""実験66: 2D の語板と、角度で何語も重ねる器（章板の模型）— 融合は 2D・3D でも同じか

着想: 2026-10-29 持ち主「単語を 1D じゃなくて 2D にして、3D（2D 板数枚）で作った場合も融合の結果は一緒？ 語板 2D → 章板 3D → 単位板（3D の融合）。
      計算量がいたずらに増えるだけならまた考えるけど、語板・章板・一単位板となるとき絶対に便利だと思う」。
      Sin-CNN3 の答え（学び 64・docs/fusion_explained.md）: 板の位相を混ぜると壊れ、出口の明るさで足すと確実。理由は次元によらない（光は e^{i 位相} で効き、何枚も重なる）。
      ただし 3D には「体積の中に角度を変えて何枚も重ね書きし、角度で選んで読む」（体積ホログラムの多重）という 1D にない形がある。
方法: 板 = 縦 32 × 横 32 の位相（2D）を 4 枚、板の間は 2D の角スペクトル法（64 × 64 に広げて回折）。出口 = 2D のレンズ（2D FFT）の決まった点（値 52 個 = 8 × 7 の点）。
      材料: 会話の名詞（実験56 と同じ数の表、非公開）、観点 5（どんな・を・が・で・に）、平らにした目標・出口のずらし（実験58 の決まり）。
  A. 2D の語板を語ごとに 1 器（2 語: 1 組ずつ、20 組）。融合: 位相の平均 / 出口の明るさの平均（並べて足す）/ 派生（2 語の平均を学んでから各語へ 200 回）+ 位相の平均。
     差 = 融合の答えと 2 語の答えの平均の全変動距離（1D の実験34・37 と同じ）。
  B. 章板の模型: 1 つの 2D の器（板 4 枚、共通）に W 語（W = 2・8・32）を重ねる。語 w・観点 v の入口 = 2D の傾いた光（横の傾き = 語、縦の傾き = 観点）。
     比べ: 語ごとに別の器（A と同じ形）。当たり（確かめ用の交差エントロピー）・値に届く光・計算量（器の数 × 板の画素）。
     融合: 2 語の入口の光を「干渉しない形」（別々の光源、明るさで足す）で同時に入れる → 答えはちょうど平均になるはず。
           「干渉する形」（同じ光源から 2 つの角度）でも入れて、干渉で答えがゆがむ大きさを測る。
結果の置き場: results/exp66.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp66_2d_goban_fusion.py（板を厚くする: LAYERS=16 WORDS=8,32 A_WORDS=32）
"""
import json, math, os, runpy, sys, time

import torch

sys.argv = [sys.argv[0], "--seeds", "0"]
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "exp56_goban_grow.py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import evaluate

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t00 = time.time()
NTR, NTE, prior, NVIEW = ns["NTR"], ns["NTE"], ns["prior"], ns["NVIEW"]


def shift(T, step=9):
    out = T.clone()
    for v in range(T.shape[1]):
        out[:, v, [(r + step * v) % 47 for r in range(47)]] = T[:, v, :47]
    return out


T, TE = shift(NTR[:400]), shift(NTE[:400]); pr = prior(shift(NTR))
TG = (T + 5 * pr[None]) / (T.sum(2, keepdim=True) + 5); NV, NX = TG.shape[1], TG.shape[2]
N, K, DZ = 32, int(os.environ.get("LAYERS", "4")), 32.0; L = 2 * N                      # 板の枚数 K（LAYERS で変える。厚い体積のつもり）                                                          # 2D の板 32 × 32 を 4 枚、64 × 64 に広げて回折
fx = torch.fft.fftfreq(L); KX, KY = torch.meshgrid(fx, fx, indexing="ij")
H = torch.exp(1j * DZ * torch.sqrt(((2 * math.pi) ** 2 - (2 * math.pi) ** 2 * (KX ** 2 + KY ** 2)).clamp(min=0)))   # 2D の角スペクトル法
SPOTS = [(8 + i // 7, 8 + i % 7) for i in range(NX)]                                        # 出口の点（2D FFT の番号）
SI = torch.tensor([s[0] for s in SPOTS]); SJ = torch.tensor([s[1] for s in SPOTS])
yy, xx = torch.meshgrid(torch.arange(N, dtype=torch.float32), torch.arange(N, dtype=torch.float32), indexing="ij")


def wave2d(a, b):
    """横に a 周、縦に b 周傾いた平らな光（2D の観点・語の入口）。"""
    return (torch.exp(1j * 2 * math.pi * (a * xx + b * yy) / N) / N).to(torch.complex64)


def device(P, u):
    """P: (..., K, N, N) の位相、u: (..., N, N) の入口の場。出口の点の明るさ (..., 値)。"""
    bs = torch.broadcast_shapes(P.shape[:-3], u.shape[:-2]); u = u.expand(*bs, N, N)      # 語と観点の形をそろえる
    u = torch.nn.functional.pad(u, (0, N, 0, N))
    for k in range(K):
        u = torch.cat([torch.cat([u[..., :N, :N] * torch.exp(1j * P[..., k, :, :]), u[..., :N, N:]], -1), u[..., N:, :]], -2)
        u = torch.fft.ifft2(torch.fft.fft2(u, norm="ortho") * H, norm="ortho")
    F = torch.fft.fft2(u[..., :N, :N], norm="ortho")
    return F.abs()[..., SI, SJ] ** 2


def fit(P, U, tg, steps, lr=0.03):
    """P を学習。U: 入口の場（P と同じ前の形 + (観点, N, N)）、tg: 目標 (…, 観点, 値)。"""
    P = torch.nn.Parameter(P.clone()); opt = torch.optim.Adam([P], lr=lr); q = torch.cat([tg, torch.zeros(*tg.shape[:-1], 1)], -1)
    for s in range(steps):
        W = device(P.unsqueeze(-4), U); p = torch.cat([W, (1 - W.sum(-1, keepdim=True)).clamp(min=1e-9)], -1)
        loss = -(q * torch.log(p + 1e-9)).sum(-1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return P.detach()


pr2 = lambda W: 0.98 * W / W.sum(-1, keepdim=True).clamp(min=1e-12) + 0.02 / W.shape[-1]
tvd = lambda a, b: float((a - b).abs().sum(-1).mean() / 2)
init = lambda *s, seed=0: math.pi * (torch.rand(*s, generator=torch.Generator().manual_seed(seed)) < 0.5).float()
UV = torch.stack([wave2d(0, v + 1) for v in range(NV)])                                      # 観点 = 縦の傾き（語ごとに別の器のとき）
res = dict(exp="66", layers=K)
# A. 2D の語板（語ごとに 1 器）の融合
t0 = time.time(); W_ = int(os.environ.get("A_WORDS", "40"))
PA = fit(init(W_, K, N, N), UV, TG[:W_], 1500)
WA = device(PA.unsqueeze(-4), UV); evA = evaluate(pr2(WA), TE[:W_])
pairs = torch.arange(W_).view(-1, 2)
avg_ans = (pr2(WA[pairs[:, 0]]) + pr2(WA[pairs[:, 1]])) / 2
Pm = torch.angle(torch.exp(1j * PA[pairs[:, 0]]) + torch.exp(1j * PA[pairs[:, 1]]))
e_phase = tvd(pr2(device(Pm.unsqueeze(-4), UV)), avg_ans)
e_bright = tvd(pr2((WA[pairs[:, 0]] + WA[pairs[:, 1]]) / 2), avg_ans)
Pmix = fit(init(len(pairs), K, N, N, seed=5), UV, (TG[pairs[:, 0]] + TG[pairs[:, 1]]) / 2, 1500)        # 2 語の平均を学んだ 1 枚
Pa = fit(Pmix, UV, TG[pairs[:, 0]], 200); Pb = fit(Pmix, UV, TG[pairs[:, 1]], 200)                       # そこから各語へ 200 回（派生）
avg_der = (pr2(device(Pa.unsqueeze(-4), UV)) + pr2(device(Pb.unsqueeze(-4), UV))) / 2
e_der = tvd(pr2(device(torch.angle(torch.exp(1j * Pa) + torch.exp(1j * Pb)).unsqueeze(-4), UV)), avg_der)
res["A"] = dict(当たり=evA["bits"], 位相の平均=e_phase, 明るさの平均=e_bright, 派生して位相の平均=e_der, sec=time.time() - t0)
print("A 2D の語板: 当たり", round(evA["bits"], 3), "融合の差: 位相の平均", round(e_phase, 3), "明るさの平均", round(e_bright, 3), "派生 + 位相の平均", round(e_der, 3), f"{time.time() - t0:.0f} 秒", flush=True)
# B. 章板の模型: 1 つの器に W 語を角度で重ねる
res["B"] = {}
for Wn in [int(x) for x in os.environ.get("WORDS", "2,8,32").split(",")]:
    t0 = time.time()
    U = torch.stack([torch.stack([wave2d(w - Wn // 2, v + 1) for v in range(NV)]) for w in range(Wn)])   # (語, 観点, N, N) 横の傾き = 語
    Ps = fit(init(K, N, N, seed=Wn), U, TG[:Wn], 1500)                                      # 共通の器 1 つ
    Wsh = device(Ps.unsqueeze(-4).unsqueeze(-4), U); ev = evaluate(pr2(Wsh), TE[:Wn])
    sep = evaluate(pr2(WA[:Wn]), TE[:Wn]) if Wn <= W_ else None
    pp = torch.arange(Wn).view(-1, 2); avg = (pr2(Wsh[pp[:, 0]]) + pr2(Wsh[pp[:, 1]])) / 2
    inc = tvd(pr2((Wsh[pp[:, 0]] + Wsh[pp[:, 1]]) / 2), avg)                                # 干渉しない 2 つの光源（明るさで足す）
    coh = tvd(pr2(device(Ps.unsqueeze(-4).unsqueeze(-4), (U[pp[:, 0]] + U[pp[:, 1]]) / math.sqrt(2))), avg)   # 同じ光源から 2 つの角度（干渉する）
    res["B"][Wn] = dict(当たり_共通の器=ev["bits"], 当たり_語ごとの器=(sep["bits"] if sep else None), 光=float(Wsh.sum(-1).mean()),
                        融合_干渉しない=inc, 融合_干渉する=coh, 器の画素=K * N * N, 語ごとの器の画素=Wn * K * N * N, sec=time.time() - t0)
    print("B", Wn, "語を 1 つの器に: 当たり", round(ev["bits"], 3), "（語ごとの器", round(sep["bits"], 3) if sep else "-", "）光", round(float(Wsh.sum(-1).mean()), 3),
          "融合: 干渉しない", round(inc, 3), "干渉する", round(coh, 3), f"{time.time() - t0:.0f} 秒", flush=True)
res["比べ_表"] = evaluate(TG[:40], TE[:40])["bits"]; res["sec"] = time.time() - t00
with open("results/exp66.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print("表（40 語）", round(res["比べ_表"], 3), f"全体 {res['sec']:.0f} 秒")
