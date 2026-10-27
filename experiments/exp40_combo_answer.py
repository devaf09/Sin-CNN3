"""実験40: 「圧力 + 熱だからこそ柔らかくなる」— どちらの道具にもない答えを、2 段の器（明るさを測って出し直す）で出せるか

着想: 学び 32 と説明（docs/explain_20261027.md）で、曲がる処理のない器は「答えの足し合わせ」しかできず、圧力だけでも熱だけでも出ない答え
      （柔らかい）は、圧力 + 熱でも出ないと説明した。これを実験で確かめ、明るさを測って出し直す 2 段の器（実験38 と同じ部品）なら出せるかを見る。
方法: 実験32 の作り（道具 = 要素の光の重ね合わせ、単語ごとの格子 1 KB、縦 100、列 100）。結果に「柔らかい」を足す。リンゴの格子だけを学習:
        プレス機（圧力）→ ジュース、鍋（熱）→ 煮る 70%・食べる 30%、包丁（刃）→ 切る、ミキサー（刃 + 回転）→ ジュース 70%・細かい 30%、
        圧力鍋（圧力 + 熱）→ 柔らかい 80%・煮る 20%（この組み合わせのときだけ柔らかい）
      1 段: 格子 1 枚（曲がる処理なし）。2 段: 格子 → 明るさを測る → √明るさの光を出し直す → 格子。3 seed、1500 回。
指標: 道具ごとの答えの表との差、圧力鍋で「柔らかい」に入った光の割合、プレス機・鍋だけで「柔らかい」に入った光の割合（出てはいけない）。
結果の置き場: results/exp40.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp40_combo_answer.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--Fw", type=int, default=8)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp40.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, L = a.N, 2 * a.N
ELEM = ["刃", "回転", "圧力", "熱"]; RES = ["ジュース", "細かい", "切る", "煮る", "食べる", "柔らかい"]
freq = {e: -40 + 8 * i for i, e in enumerate(ELEM)}; freq.update({r: 6 + 6 * i for i, r in enumerate(RES)})
TOOL = {"プレス機": ["圧力"], "鍋": ["熱"], "包丁": ["刃"], "ミキサー": ["刃", "回転"], "圧力鍋": ["圧力", "熱"]}
TAB = {"プレス機": {"ジュース": 1}, "鍋": {"煮る": .7, "食べる": .3}, "包丁": {"切る": 1}, "ミキサー": {"ジュース": .7, "細かい": .3}, "圧力鍋": {"柔らかい": .8, "煮る": .2}}
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, a.Fw + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)
tool_light = lambda els: sum(wave(freq[e]) for e in els) / math.sqrt(len(els))
rb = torch.tensor([freq[r] % N for r in RES]); grid = lambda A: sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))


def stage(A, u):
    G = grid(A); u = torch.nn.functional.pad(u, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return u[:, :N]


def run(As, u):
    u = stage(As[0], u)
    if len(As) == 2:
        I = u.abs() ** 2; u = torch.sqrt(I / I.sum(1, keepdim=True) + 1e-12).to(torch.complex64)   # 明るさを測って出し直す
        u = stage(As[1], u)
    return torch.fft.fft(u, n=N, dim=-1, norm="ortho").abs()[:, rb] ** 2


tools = list(TOOL); u0 = torch.stack([tool_light(TOOL[t]) for t in tools])
q = torch.zeros(len(tools), len(RES) + 1)
for i, t in enumerate(tools):
    for r, p in TAB[t].items():
        q[i, RES.index(r)] = p
res = dict(exp="40", args=vars(a), rows=[]); soft = RES.index("柔らかい")
for kind in ("1 段（曲がる処理なし）", "2 段（明るさを測って出し直す）"):
    for seed in range(a.seeds):
        t0 = time.time(); g = torch.Generator().manual_seed(seed)
        As = [torch.nn.Parameter(0.1 * torch.randn(4, a.Fw, a.Fw, generator=g)) for _ in range(1 if kind.startswith("1") else 2)]
        opt = torch.optim.Adam(As, lr=0.01)
        for step in range(a.steps):
            W = run(As, u0); p = torch.cat([W, (W.new_ones(len(W)) - W.sum(1)).clamp(min=1e-9)[:, None]], 1) if len(As) == 1 else torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
            loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W = run(As, u0); D = (W / W.sum(1, keepdim=True)).numpy()
            r = dict(kind=kind, seed=seed, tv={t: float(np.abs(D[i] - q[i, :-1].numpy()).sum() / 2) for i, t in enumerate(tools)},
                     soft={t: float(D[i, soft]) for i, t in enumerate(tools)}, sec=time.time() - t0)
        res["rows"].append(r)
        print(f"{kind} seed {seed}: 柔らかいの割合 圧力鍋 {100 * r['soft']['圧力鍋']:.0f}%・プレス機 {100 * r['soft']['プレス機']:.0f}%・鍋 {100 * r['soft']['鍋']:.0f}%  "
              f"表との差 " + " ".join(f"{t}{v:.2f}" for t, v in r["tv"].items()) + f"  {r['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
