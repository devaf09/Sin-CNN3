"""実験64: 融合しやすい語板 — 2 段の器の真ん中（明るさの段）に語ごとの「透過の模様」を置き、そこで混ぜる

着想: 2026-10-29 持ち主「融合が重要になる。非線形のやり方を変えるとか、融合できるとかないかなぁ」。
      これまで: 板の位相を混ぜると壊れる（学び 34・36）。明るさで足すのは確実（並べて足す）。
      「明るさを測って出し直す」段（実験38・40）では、情報が明るさ（足し算が効く形）になっている。
      そこで器を 3 つに分ける:
        共通の板 A（全部の語で同じ）→ 明るさ I を測る → 語ごとの透過の模様 m（0〜1 の 64 個、明るさを何割通すか）→
        明るさに比例した光を出し直す（画素ごとに別々の光源 = 干渉しない、インコヒーレント）→ 共通の板 B → 出口
      出し直した光が互いに干渉しないなら、出口の明るさ = Σ_画素 |B の伝わり方|² × m × I（m について 1 次 = 線形）。
      → **2 つの語の模様 m を平均すると、出口の答えもちょうど 2 つの答えの平均になるはず**（板の位相を混ぜるのと違って、壊れない）。
      語の知識は m（64 個の数）だけ。板 A・B は全部の語で共有。
      変種（環境変数 PER_VIEW=1）: 観点ごとに別の模様 m（5 × 64 個）。観点ごとに板の別の場所を通すつもり。
方法: 実験58 と同じ材料・名詞 400 語（会話、非公開）・平らにした目標・出口のずらし。
      比べ: 語ごとの位相の語板（実験58、1 語 = 位相 512 個）。
      確かめ: (1) 当たり（確かめ用の交差エントロピー）、(2) 融合: でたらめな 300 組の 2 語について、模様の平均の答えと、2 語の目標の平均との差（全変動距離）。
              位相の語板では、位相の平均（板を混ぜる）と比べる。(3) 引き算: 融合した模様から片方を引いて、もう片方に戻るか。
結果の置き場: results/exp64.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp64_incoherent_fusion.py
"""
import json, math, os, runpy, sys, time

import torch

sys.argv = [sys.argv[0], "--seeds", "0"]
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "exp56_goban_grow.py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate
from grain import N, K, DZ, view_wave

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t00 = time.time()
NTR, NTE, prior, FREQ, NVIEW = ns["NTR"], ns["NTE"], ns["prior"], ns["FREQ"], ns["NVIEW"]


def shift(T, step=9):
    out = T.clone()
    for v in range(T.shape[1]):
        out[:, v, [(r + step * v) % 47 for r in range(47)]] = T[:, v, :47]
    return out


T, TE = shift(NTR[FREQ]), shift(NTE[FREQ]); pr = prior(shift(NTR))
tg = (T + 5 * pr[None]) / (T.sum(2, keepdim=True) + 5); NW, NVw, NX = tg.shape
L = 2 * N; H = torch.exp(1j * DZ * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2)); VAL0 = 10


def plates(u, P):
    u = torch.nn.functional.pad(u, (0, L - N))
    for k in range(K):
        u = torch.cat([u[..., :N] * torch.exp(1j * P[:, k]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
    return u[..., :N]


u0 = view_wave(range(1, NVw + 1)); eye = torch.eye(N, dtype=torch.complex64)


def model(PA, PB, M):
    """M: (語, N) または (語, 観点, N) の透過の模様（0〜1）。出口の明るさ (語, 観点, 値)。"""
    I = plates(u0, PA).abs() ** 2                                                            # (観点, N) 共通の板 A の明るさ（語によらない）
    Kb = torch.fft.fft(plates(eye, PB), dim=-1, norm="ortho").abs()[:, VAL0:VAL0 + NX] ** 2   # (画素, 値) 画素 1 つから出した光が各出口に届く明るさ
    if M.dim() == 2:
        return torch.einsum("wn,vn,nx->wvx", M, I, Kb)                                      # 干渉しないので足し算（線形）
    return torch.einsum("wvn,vn,nx->wvx", M, I, Kb)                                         # 観点ごとに別の模様（観点 = 板の別の場所）


res = []
import argparse
PER_VIEW = os.environ.get("PER_VIEW") == "1"
for seed in range(2):
    g = torch.Generator().manual_seed(seed)
    PA = torch.nn.Parameter(math.pi * (torch.rand(N, K, generator=g) < 0.5).float()); PB = torch.nn.Parameter(math.pi * (torch.rand(N, K, generator=g) < 0.5).float())
    Z = torch.nn.Parameter(torch.zeros(NW, NVw, N) if PER_VIEW else torch.zeros(NW, N)); opt = torch.optim.Adam([PA, PB, Z], lr=0.03)
    q = torch.cat([tg, torch.zeros(NW, NVw, 1)], -1)
    for step in range(2000):
        W = model(PA, PB, torch.sigmoid(Z)); p = torch.cat([W, (1 - W.sum(-1, keepdim=True)).clamp(min=1e-9)], -1)
        loss = -(q * torch.log(p + 1e-9)).sum(-1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        M = torch.sigmoid(Z); W = model(PA, PB, M); ev = evaluate(prob(W), TE); light = float(W.sum(-1).mean())
        gen = torch.Generator().manual_seed(100 + seed); pairs = torch.randint(0, NW, (300, 2), generator=gen)
        tvd = lambda a, b: float((prob(a) - b).abs().sum(-1).mean() / 2)
        fuse = model(PA, PB, (M[pairs[:, 0]] + M[pairs[:, 1]]) / 2)
        f_err = tvd(fuse, (prob(W[pairs[:, 0]]) + prob(W[pairs[:, 1]])) / 2)                # 融合の答え と 2 語の答えの平均 の差
        f_err_t = tvd(fuse, (tg[pairs[:, 0]] + tg[pairs[:, 1]]) / 2)                        # 2 語の目標の平均との差
        back = model(PA, PB, (2 * (M[pairs[:, 0]] + M[pairs[:, 1]]) / 2 - M[pairs[:, 1]]).clamp(0, 1))   # 融合から B を引いて A に戻す
        s_err = tvd(back, prob(W[pairs[:, 0]]))
    # 比べ: 語ごとの位相の語板（実験58）と、位相の平均による融合
    P = train(tg, torch.ones(tg.shape[:2], dtype=torch.bool), 2000, seed, lr=0.03); Wg = run(P, NVw, NX)
    with torch.no_grad():
        evg = evaluate(prob(Wg), TE)
        ang = lambda x: torch.angle(torch.exp(1j * x))
        Pm = torch.angle(torch.exp(1j * P[pairs[:, 0]]) + torch.exp(1j * P[pairs[:, 1]]))       # 位相の平均（円の上で）
        g_err = tvd(run(Pm, NVw, NX), (prob(Wg[pairs[:, 0]]) + prob(Wg[pairs[:, 1]])) / 2)
        g_err_t = tvd(run(Pm, NVw, NX), (tg[pairs[:, 0]] + tg[pairs[:, 1]]) / 2)
    row = dict(exp="64", seed=seed, 透過の模様={"確かめ用": ev, "光": light, "融合の差（答えの平均と）": f_err, "融合の差（目標の平均と）": f_err_t, "引き算で戻す差": s_err, "1 語の数": N * (NVw if PER_VIEW else 1), "観点ごとの模様": PER_VIEW},
               位相の語板={"確かめ用": evg, "融合の差（答えの平均と）": g_err, "融合の差（目標の平均と）": g_err_t, "1 語の数": N * K},
               表=evaluate(tg, TE), sec=time.time() - t00)
    res.append(row)
    with open("results/exp64.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(seed, "透過の模様:", {k: (round(v["bits"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in row["透過の模様"].items()},
          "| 位相の語板:", {k: (round(v["bits"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in row["位相の語板"].items()}, "| 表", round(row["表"]["bits"], 3), flush=True)
