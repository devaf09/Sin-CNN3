"""実験34: 融合のいろいろな形（優先度低め、持ち主の依頼）— リンゴの格子とトマトの格子を、どう合わせると何が起きるか

着想: 2026-10-27 持ち主の依頼「融合について、優先度低めの実験で、いろいろパターンを変えてやってみたい」。実験31 で、位相を平均すると壊れ、
      2 枚に光を分けて足すとちょうど平均になった。ほかの形も並べて、2 つの意味のどれだけ近くに出るか・光がどれだけ残るかを比べる。
方法: 実験31 と同じ（物 = 単語ごとの格子 1 KB、入口は「何か」× 観点、形 1）。リンゴとトマトの格子を学習してから、8 つの形で合わせる:
        1 並べて足す（光を半分ずつ 2 枚に通し、出口で足す。a = 明るさを足す = 干渉しない、b = 場を足す = 干渉する）     2 比率 80 : 20 で並べて明るさを足す
        3 差分（トマトの側だけ位相を半周ずらして足す）          4 積み重ねる（リンゴの 100 列のあとにトマトの 100 列）
        5 列ごとに交互（偶数列リンゴ・奇数列トマト）            6 位相の平均
        7 複素数の板（各列の透け具合を (e^(iG_A) + e^(iG_B))/2 にする。光は減る）
        8 学習でまとめ直す（位相の平均から始め、両方の答えの平均に合うよう 300 回学習）
指標: 目標（1・3〜8 は平均、2 は 80 : 20 の重み付き平均）との差（全変動距離）、入れた光のうち値に届いた割合。
      差分（3）は、目標との差の代わりに「リンゴとトマトの答えの差（リンゴ − トマト）」との相関も測る。3 seed。
結果の置き場: results/exp34.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp34_fusion.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--Fw", type=int, default=8)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp34.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp21_viewpoint.py"), encoding="utf-8").read().split("res = dict(")[0]
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)
VAL, VIEW, TABLE, freq, N, L, H = (ns[k] for k in ("VAL", "VIEW", "TABLE", "freq", "N", "L", "H"))
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, a.Fw + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)
views = list(VIEW); probe = torch.stack([wave(VIEW[v]) for v in views]); vb = torch.tensor([freq[v] % N for v in VAL])


def field(T, u):
    """透け具合 T（縦 × 列、複素数）を列ごとに通した出口の場（周波数に分ける前）。"""
    u = torch.nn.functional.pad(u, (0, L - N))
    for c in range(T.shape[1]):
        u = torch.cat([u[:, :N] * T[:, c], u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return u[:, :N]


spec = lambda u: torch.fft.fft(u, n=N, dim=-1, norm="ortho").abs()[:, vb] ** 2          # 値ごとの明るさ
grid = lambda A: sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
target_of = lambda o: torch.tensor([[TABLE[o][v].get(x, 0.0) for x in VAL] for v in views])


def train(A, tgt, steps):
    opt = torch.optim.Adam([A], lr=0.01)
    for step in range(steps):
        W = spec(field(torch.exp(1j * grid(A)), probe)); p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
        q = torch.cat([tgt, torch.zeros(len(tgt), 1)], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()


res = dict(exp="34", args=vars(a), rows=[])
dist = lambda W, t: float((W / W.sum(1, keepdim=True) - t).abs().sum(1).mean() / 2)
for seed in range(a.seeds):
    t0 = time.time(); g = torch.Generator().manual_seed(seed); As = {}
    for o in ("リンゴ", "トマト"):
        As[o] = torch.nn.Parameter(0.1 * torch.randn(4, a.Fw, a.Fw, generator=g)); train(As[o], target_of(o), a.steps)
    with torch.no_grad():
        GA, GB = grid(As["リンゴ"]), grid(As["トマト"]); TA, TB = torch.exp(1j * GA), torch.exp(1j * GB)
        fA, fB = field(TA, probe), field(TB, probe); PA, PB = spec(fA), spec(fB)
        dA, dB = PA / PA.sum(1, keepdim=True), PB / PB.sum(1, keepdim=True); mix, mix82 = (dA + dB) / 2, 0.8 * dA + 0.2 * dB
        alt = torch.where((torch.arange(TA.shape[1]) % 2 == 0)[None, :].expand_as(TA), TA, TB)      # 偶数列はリンゴ、奇数列はトマト
        out = {"1a 並べて明るさを足す（干渉しない）": 0.5 * PA + 0.5 * PB,
               "1b 並べて場を足す（干渉する）": spec(math.sqrt(.5) * fA + math.sqrt(.5) * fB),
               "2 比率 80 : 20 で並べて足す": 0.8 * PA + 0.2 * PB,
               "3 差分（半周ずらして足す）": spec(math.sqrt(.5) * fA - math.sqrt(.5) * fB),
               "4 積み重ねる": spec(field(torch.cat([TA, TB], 1), probe)),
               "5 列ごとに交互": spec(field(alt, probe)),
               "6 位相の平均": spec(field(torch.exp(1j * (GA + GB) / 2), probe)),
               "7 複素数の板": spec(field((TA + TB) / 2, probe))}
    A8 = torch.nn.Parameter(((As["リンゴ"] + As["トマト"]) / 2).detach().clone()); train(A8, (target_of("リンゴ") + target_of("トマト")) / 2, 300)
    with torch.no_grad():
        out["8 学習でまとめ直す"] = spec(field(torch.exp(1j * grid(A8)), probe))
        row = dict(seed=seed, sec=time.time() - t0, forms={})
        for k, W in out.items():
            t = mix82 if k.startswith("2") else mix
            light = float(W.sum(1).mean())
            r = dict(dist=dist(W, t), light=light)
            if k.startswith("3"):                                                         # 差分: 明るさがリンゴとトマトの答えの差とどれだけ相関するか
                dd = (dA - dB).abs().flatten().numpy(); ww = (W / W.sum(1, keepdim=True)).flatten().numpy()
                r["corr_diff"] = float(np.corrcoef(dd, ww)[0, 1])
            row["forms"][k] = r
    res["rows"].append(row)
    print(f"seed {seed}（{row['sec']:.0f} 秒）: " + "  ".join(f"{k} 差 {v['dist']:.3f}・光 {100 * v['light']:.0f}%" + (f"・差との相関 {v['corr_diff']:.2f}" if "corr_diff" in v else "") for k, v in row["forms"].items()), flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
