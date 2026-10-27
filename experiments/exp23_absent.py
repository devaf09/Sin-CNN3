"""実験23: 存在しない属性（拡散か「なし」か）と、まちまちな属性 — 観点周波数の続き

着想: 2026-10-27 持ち主の案「存在しない属性のときに拡散か、存在しないを選択しないといけない（リンゴの目の色、サイズがまちまちの船のサイズ感）。
      拡散にも意味を持たせたいから、無理にまとめなくてもいい。後で融合の差分で使うかも」。
方法: 実験21 と同じ作り（記号 = 周波数の波、観点 = 縦の周波数のずらし、順引きを学習、値の光を逆向きに通して逆引き、観点ごとに掛け算）。
      縦 128 画素、記号は周波数 −63, −56, …, 63（7 の倍数、19 個）、観点 6 つ（ずらし 1〜6）。
      物 4: リンゴ・トマト・猫・船。値 14: 赤 緑 青 黄 黒 小 中 大 丸い 細長い 甘い 酸っぱい しゃきしゃき なし（「なし」は 2 つ目の作り方でだけ使う）。
      存在しない属性（リンゴ・トマト・船の目の色、猫・船の味と触感）の答えを 2 とおりで学習:
        拡散: その観点で使う値の全部に均等に散らす（「分からない・関係ない」）
        なし: 「なし」の記号 1 つに集める（「存在しない」）
      まちまちな属性: 船の大きさ = 小 20%・中 30%・大 50%、猫の大きさ = 小 70%・中 30%。
指標: 存在しない属性の出口の広がり（エントロピー = 散らばり具合。均等なら最大）、問い合わせの点数（2 とおりの違い）。器は実験21 と同じ、1500 回、3 seed。
結果の置き場: results/exp23.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp23_absent.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=128)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp23.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
OBJ = ["リンゴ", "トマト", "猫", "船"]
VAL = ["赤", "緑", "青", "黄", "黒", "小", "中", "大", "丸い", "細長い", "甘い", "酸っぱい", "しゃきしゃき", "なし"]
SYM = OBJ + VAL; freq = {s: -63 + 7 * i for i, s in enumerate(SYM)}
VIEW = {"色": 1, "大きさ": 2, "形": 3, "味": 4, "触感": 5, "目の色": 6}
VALS_OF = {"色": ["赤", "緑", "青", "黄", "黒"], "大きさ": ["小", "中", "大"], "形": ["丸い", "細長い"], "味": ["甘い", "酸っぱい"],
           "触感": ["しゃきしゃき"], "目の色": ["青", "黄", "黒"]}
ABSENT = None                                                                          # 存在しない属性の印
TABLE = {"リンゴ": {"色": {"赤": .8, "緑": .2}, "大きさ": {"小": .3, "中": .7}, "形": {"丸い": 1}, "味": {"甘い": .6, "酸っぱい": .4}, "触感": {"しゃきしゃき": 1}, "目の色": ABSENT},
         "トマト": {"色": {"赤": .9, "緑": .1}, "大きさ": {"小": .6, "中": .4}, "形": {"丸い": 1}, "味": {"甘い": .3, "酸っぱい": .7}, "触感": ABSENT, "目の色": ABSENT},
         "猫": {"色": {"黒": .4, "黄": .3, "赤": .3}, "大きさ": {"小": .7, "中": .3}, "形": {"細長い": .6, "丸い": .4}, "味": ABSENT, "触感": ABSENT, "目の色": {"青": .3, "黄": .5, "黒": .2}},
         "船": {"色": {"黒": .3, "青": .4, "赤": .3}, "大きさ": {"小": .2, "中": .3, "大": .5}, "形": {"細長い": 1}, "味": ABSENT, "触感": ABSENT, "目の色": ABSENT}}
QUERIES = {"目の色が青い": {"目の色": "青"}, "赤くて丸い": {"色": "赤", "形": "丸い"},
           "大きくて細長い": {"大きさ": "大", "形": "細長い"}, "赤くて目の色が黄色": {"色": "赤", "目の色": "黄"}}
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)


def target(o, v, mode):
    t = TABLE[o][v]
    if t is ABSENT:
        return {x: 1 / len(VALS_OF[v]) for x in VALS_OF[v]} if mode == "拡散" else {"なし": 1.0}
    return t


def grid(A):
    return sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))


def forward(A, u0):
    G = grid(A); u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return u


def backward(A, uL):
    G = grid(A); u = uL
    for col in reversed(range(N)):
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * torch.conj(H), n=L, dim=-1, norm="ortho")
        u = torch.cat([u[:, :N] * torch.exp(-1j * G[:, col]), u[:, N:]], 1)
    return u[:, :N]


spec = lambda u: torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs() ** 2
bins = torch.tensor([freq[s] % N for s in SYM])
pairs = [(o, v) for o in OBJ for v in VIEW]
u0 = torch.stack([wave(freq[o] + VIEW[v]) for o, v in pairs])
res = dict(exp="23", args=vars(a), rows=[])
for mode in ("拡散", "なし"):
    q = torch.zeros(len(pairs), len(SYM) + 1)
    for n, (o, v) in enumerate(pairs):
        for val, p in target(o, v, mode).items():
            q[n, SYM.index(val)] = p
    for seed in range(a.seeds):
        t0 = time.time(); g = torch.Generator().manual_seed(seed)
        A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
        for step in range(a.steps):
            W = spec(forward(A, u0))[:, bins]
            p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
            loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W = spec(forward(A, u0))[:, bins]; D = W / W.sum(1, keepdim=True)
            tv = float((D - q[:, :-1]).abs().sum(1).mean() / 2)
            ent = {f"{o}の{v}": float(-(D[n] * torch.log(D[n] + 1e-12)).sum()) for n, (o, v) in enumerate(pairs) if TABLE[o][v] is ABSENT or o == "船" and v == "大きさ"}
            back = {}
            for v, d in VIEW.items():
                for val in VAL:
                    S = spec(backward(A, torch.nn.functional.pad(wave(freq[val])[None], (0, L - N))))[0]
                    back[(v, val)] = [float(S[(freq[o] + d) % N]) for o in OBJ]
            qres = {}
            for name, ev in QUERIES.items():
                sc = np.prod([back[(v, val)] for v, val in ev.items()], axis=0)
                tr = np.prod([[target(o, v, mode).get(val, 0) for o in OBJ] for v, val in ev.items()], axis=0)
                qres[name] = dict(score=(sc / sc.sum()).tolist(), truth=(tr / max(tr.sum(), 1e-12)).tolist())
        r = dict(mode=mode, seed=seed, tv=tv, entropy=ent, queries=qres, sec=time.time() - t0); res["rows"].append(r)
        print(f"{mode} seed {seed}: 表との差 {tv:.3f}  広がり（エントロピー）" + " ".join(f"{k} {v:.2f}" for k, v in ent.items()) + f"  {r['sec']:.0f} 秒")
        for name, x in qres.items():
            print(f"   「{name}」 光の器 " + " ".join(f"{o}{100 * s:.0f}" for o, s in zip(OBJ, x["score"])) + "  表 " + " ".join(f"{o}{100 * s:.0f}" for o, s in zip(OBJ, x["truth"])))
res.update(OBJ=OBJ, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
