"""実験22: 鋭さを上げる方法の比べ — 2 つの記号が重なった光から、明るい方だけを残す（何回も通す）

着想: 実験20 で、学習した光の格子 + 飽和は 1 回通すと鋭さが +2〜4 点だった。持ち主「ほかにも案があれば試して OK」。
      記号は周波数の波なので、レンズ（フーリエ変換）を通すと記号ごとの光の点に分かれる。その点の面に光の部品を置けば、記号ごとの明るさに直接働きかけられる。
方法: 入力 = 2 つの記号の重ね合わせ（勝者の明るさの割合 s = 0.55〜0.9、位相でたらめ）。1 回 = レンズ → 部品 → 逆のレンズ → 明るさの合計を 1 に戻す（増幅）。これを k = 1〜5 回。
      部品（光の回路で作れるもの）:
        測って出し直す（2 乗）: 点ごとに明るさ I を受光素子で測り、明るさ I² の光を出し直す（前のプロジェクトの「測って出し直す」非線形と同じ仕組み）
        可飽和吸収体: 明るいほど通りやすい。通る割合 1 − α/(1 + I/Is)（α = 0.9、Is = 平均の明るさ）
        リミッタ（実験14・20 の「飽和」、2 光子吸収も同じ形）: 明るいほど通りにくい。I/(1 + I/Is)
      比較相手: 実験20 の学習した格子 + 飽和（s = 0.003）を k 回通す。
式の予想（明るさの割合 s を 1 回ごとに）: 2 乗 → s²/(s² + (1 − s)²)。可飽和吸収体・リミッタも、点ごとの明るさの式から計算できる（下の formula）。
指標: 勝者の割合の平均（入口 → k 回目）、式との差。乱数（入力）は 3 seed。
結果の置き場: results/exp22.txt。図は tools/fig_exp22.py → docs/images/exp22_sharpen.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp22_sharpen.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--learned", type=int, default=1, help="1 なら実験20 の学習した格子 + 飽和も比べる（1 seed あたり 1〜2 分）")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp22.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, K = a.N, 6
pos = torch.arange(N, dtype=torch.float32)
freq = torch.tensor([-10 + 2 * i for i in range(K)]); bins = freq % N
waves = (torch.exp(1j * 2 * math.pi * freq[:, None].float() * pos / N) / math.sqrt(N)).to(torch.complex64)
ELEM = {"測って出し直す（2 乗）": lambda I, m: I ** 2,
        "可飽和吸収体（明るいほど通る）": lambda I, m: I * (1 - 0.9 / (1 + I / m)),
        "リミッタ・2 光子吸収（明るいほど通らない）": lambda I, m: I / (1 + I / m)}


def formula(s, name):
    """2 つの点の明るさ s, 1 − s に部品を 1 回かけ、割合に戻す（式）。平均の明るさ m は点 2 つぶんの合計 / 点の数ではなく、2 点の平均とする。"""
    w, l = s, 1 - s; m = 0.5
    w2, l2 = ELEM[name](w, m), ELEM[name](l, m)
    return w2 / (w2 + l2)


def one_pass(u, name):
    U = torch.fft.fft(u, n=N, dim=-1, norm="ortho")                                     # レンズ: 記号ごとの点に分かれる
    I = U.abs() ** 2; m = I[:, bins].sum(1, keepdim=True) / 2                           # 2 点の平均の明るさ
    T = ELEM[name](I, m) / (I + 1e-12)                                                   # 明るさが何倍になるか
    U = U * torch.sqrt(T)                                                                # 振幅は √倍（位相はそのまま）
    u = torch.fft.ifft(U, n=N, dim=-1, norm="ortho")                                    # 逆のレンズで波に戻す
    return u / (u.abs() ** 2).sum(1, keepdim=True).sqrt()                                # 明るさの合計を 1 に戻す（増幅）


def share(u, pair):
    I = torch.fft.fft(u, n=N, dim=-1, norm="ortho").abs() ** 2
    w, l = I[torch.arange(len(u)), bins[pair[:, 0]]], I[torch.arange(len(u)), bins[pair[:, 1]]]
    return w / (w + l)


res = dict(exp="22", args=vars(a), rows=[])
for name in ELEM:
    curves, errs = [], []
    for seed in range(a.seeds):
        g = torch.Generator().manual_seed(seed)
        pair = torch.stack([torch.randperm(K, generator=g)[:2] for _ in range(400)]); s = 0.55 + 0.35 * torch.rand(400, generator=g)
        amp = torch.stack([s.sqrt(), (1 - s).sqrt()], 1) * torch.exp(2j * math.pi * torch.rand(400, 2, generator=g))
        u = (amp[:, :, None] * waves[pair]).sum(1); sf = s.double().clone(); c = [float(share(u, pair).mean())]; e = 0.0
        for k in range(a.k):
            u = one_pass(u, name); sf = formula(sf, name); sh = share(u, pair).double()
            c.append(float(sh.mean())); e = max(e, float((sh - sf).abs().max()))
        curves.append(c); errs.append(e)
    r = dict(kind=name, curve=np.mean(curves, 0).tolist(), curve_se=(np.std(curves, 0, ddof=1) / math.sqrt(a.seeds)).tolist(), err_formula=max(errs))
    res["rows"].append(r)
    print(f"{name:<22} 勝者の割合 " + " → ".join(f"{100 * v:.1f}" for v in r["curve"]) + f" %  式との差 {r['err_formula']:.1e}", flush=True)

if a.learned:                                                                             # 実験20 の学習した格子 + 飽和を k 回通す
    sys.argv = [sys.argv[0], "--kinds", "zzz"]
    src = open(os.path.join(os.path.dirname(__file__), "exp20_winner.py"), encoding="utf-8").read().split("res = dict(")[0]
    ns = {}; exec(src, ns)
    curves = []
    for seed in range(a.seeds):
        g = torch.Generator().manual_seed(seed)
        (ux, px, sx) = ns["data"](400, g); (vx, pv, sv) = ns["data"](400, g)
        A = torch.nn.Parameter(0.1 * torch.randn(4, 16, 16, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
        kind = "飽和 s=0.003"
        for step in range(600):
            W = ns["run"](kind, A, ux); tot = (ux.abs() ** 2).sum(1)
            p = torch.cat([W, (tot - W.sum(1)).clamp(min=1e-9)[:, None]], 1) / tot[:, None]
            loss = -torch.log(p[torch.arange(len(W)), px[:, 0]] + 1e-9).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((ns["sy"], ns["cy"]), (ns["sy"], ns["sy"]), (ns["cy"], ns["cy"]), (ns["cy"], ns["sy"]))))
            u = vx; c = [float(share(u, pv).mean())]
            for k in range(a.k):
                v = torch.nn.functional.pad(u, (0, 2 * N - N))
                for col in range(N):
                    v = torch.cat([v[:, :N] * torch.exp(1j * G[:, col]), v[:, N:]], 1)
                    v = torch.fft.ifft(torch.fft.fft(v, n=2 * N, dim=-1, norm="ortho") * ns["H"], n=2 * N, dim=-1, norm="ortho")
                    v = v / torch.sqrt(1 + v.abs() ** 2 / 0.003)
                u = v[:, :N]; u = u / (u.abs() ** 2).sum(1, keepdim=True).sqrt(); c.append(float(share(u, pv).mean()))
        curves.append(c)
    r = dict(kind="学習した格子 + 飽和（実験20）", curve=np.mean(curves, 0).tolist(), curve_se=(np.std(curves, 0, ddof=1) / math.sqrt(a.seeds)).tolist(), err_formula=None)
    res["rows"].append(r)
    print(f"{r['kind']:<22} 勝者の割合 " + " → ".join(f"{100 * v:.1f}" for v in r["curve"]) + " %", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
