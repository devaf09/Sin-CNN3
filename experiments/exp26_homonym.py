"""実験26: 2 つの意味を持つ言葉 — 果物のリンゴと人物のリンゴさん（重ね合わせ + 手がかりの掛け算）

着想: 2026-10-27 持ち主の案「文章中に同じ文字列で 2 つの意味を持つ単語に意味があるかも。果物のリンゴと人物のリンゴさんとか。そうなると確率は掛け算かなぁ」。
      見立て: 文字列「リンゴ」は 2 つの意味の光を、使われやすさの割合で重ねたもの（足し算 = 重ね合わせ、実験17 の枝分かれ）。
      周りの手がかり（食べた・笑った…）の確からしさを意味ごとに掛け算すると、一方の意味に決まる。
      式: 意味の点数 ∝ 使われやすさ(意味 | 文字列) × Π 確からしさ(手がかり | 意味)。
方法: 1 つの器に 2 種類の対応を学習させる（実験16・21 と同じ作り。記号 = 周波数の波、曲がる処理なし）。
        ずらし 0（意味の観点）: 文字列 → 意味の分布。「リンゴ」→ 果物のリンゴ 70%・人のリンゴ 30%、「ミカン」→ 果物のミカン、「太郎」→ 人の太郎。
        ずらし 1〜3（行い・味・性格）: 意味 → 値の分布（私が決めた表 TABLE。人の味・果物の性格は「拡散」= 分からない）。
      推し量り: 文字列を順向きに通して意味の明るさ（使われやすさ）を読み、手がかりごとに値の光を逆向きに通して意味の明るさ（確からしさ）を読み、掛け算する。
      比較相手: 表から計算した点数。1500 回、3 seed。
結果の置き場: results/exp26.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp26_homonym.py
"""

import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp26.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
WORD = ["「リンゴ」", "「ミカン」", "「太郎」"]
OBJ = ["果物のリンゴ", "人のリンゴ", "果物のミカン", "人の太郎"]
VAL = ["食べた", "切った", "話した", "笑った", "歩いた", "甘い", "酸っぱい", "優しい", "元気"]
SYM = WORD + OBJ + VAL; freq = {s: -48 + 6 * i for i, s in enumerate(SYM)}
VIEW = {"行い": 1, "味": 2, "性格": 3}
VALS_OF = {"行い": ["食べた", "切った", "話した", "笑った", "歩いた"], "味": ["甘い", "酸っぱい"], "性格": ["優しい", "元気"]}
SENSE = {"「リンゴ」": {"果物のリンゴ": .7, "人のリンゴ": .3}, "「ミカン」": {"果物のミカン": 1}, "「太郎」": {"人の太郎": 1}}
D = None                                                                                  # 拡散（分からない）
TABLE = {"果物のリンゴ": {"行い": {"食べた": .6, "切った": .4}, "味": {"甘い": .6, "酸っぱい": .4}, "性格": D},
         "人のリンゴ": {"行い": {"話した": .4, "笑った": .4, "歩いた": .2}, "味": D, "性格": {"優しい": .7, "元気": .3}},
         "果物のミカン": {"行い": {"食べた": .8, "切った": .2}, "味": {"甘い": .5, "酸っぱい": .5}, "性格": D},
         "人の太郎": {"行い": {"話した": .3, "笑った": .3, "歩いた": .4}, "味": D, "性格": {"優しい": .3, "元気": .7}}}
for o in TABLE:
    for v in TABLE[o]:
        if TABLE[o][v] is D:
            TABLE[o][v] = {x: 1 / len(VALS_OF[v]) for x in VALS_OF[v]}
QUERIES = {"リンゴ（手がかりなし）": ("「リンゴ」", {}), "リンゴを食べた": ("「リンゴ」", {"行い": "食べた"}), "リンゴが笑った": ("「リンゴ」", {"行い": "笑った"}),
           "リンゴは優しい": ("「リンゴ」", {"性格": "優しい"}), "リンゴは甘い": ("「リンゴ」", {"味": "甘い"}),
           "リンゴが歩いた、元気だ": ("「リンゴ」", {"行い": "歩いた", "性格": "元気"})}
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)


def grid(A):
    return sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))


def forward(A, u0):
    G = grid(A); u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return u


def backward(A, uL):
    """出口の場 (n, L) を逆向きに通す: 各段の逆（回折の逆 → 板の逆）を逆の順に。入口の場 (n, N) を返す。"""
    G = grid(A); u = uL
    for col in reversed(range(N)):
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * torch.conj(H), n=L, dim=-1, norm="ortho")
        u = torch.cat([u[:, :N] * torch.exp(-1j * G[:, col]), u[:, N:]], 1)
    return u[:, :N]


spec = lambda u: torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs() ** 2
pairs = [(w, "意味") for w in WORD] + [(o, v) for o in OBJ for v in VIEW]
u0 = torch.stack([wave(freq[o] + (VIEW[v] if v in VIEW else 0)) for o, v in pairs])
q = torch.zeros(len(pairs), len(SYM) + 1)
for n, (o, v) in enumerate(pairs):
    for val, p in (SENSE[o] if v == "意味" else TABLE[o][v]).items():
        q[n, SYM.index(val)] = p
bins = torch.tensor([freq[s] % N for s in SYM])
res = dict(exp="26", args=vars(a), rows=[])
for seed in range(a.seeds):
    t0 = time.time(); g = torch.Generator().manual_seed(seed)
    A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
    for step in range(a.steps):
        W = spec(forward(A, u0))[:, bins]
        p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        W = spec(forward(A, u0))[:, bins]; D = W / W.sum(1, keepdim=True)                   # 順引き: 値の分布
        tv = float((D - q[:, :-1]).abs().sum(1).mean() / 2); light = float(W.sum(1).mean())
        back = {}
        for v, d in VIEW.items():
            for val in VALS_OF[v]:
                S = spec(backward(A, torch.nn.functional.pad(wave(freq[val])[None], (0, L - N))))[0]
                back[(v, val)] = np.array([float(S[(freq[o] + d) % N]) for o in OBJ])
        qres = {}
        for name, (w, ev) in QUERIES.items():
            prior = np.array([float(W[WORD.index(w), SYM.index(o)]) for o in OBJ])            # 順向き: 文字列 → 意味の明るさ（使われやすさ）
            sc = prior * np.prod([back[(v, val)] for v, val in ev.items()], axis=0) if ev else prior
            tr = np.array([SENSE[w].get(o, 0) for o in OBJ]) * (np.prod([[TABLE[o][v].get(val, 0) for o in OBJ] for v, val in ev.items()], axis=0) if ev else 1)
            qres[name] = dict(score=(sc / sc.sum()).tolist(), truth=(tr / tr.sum()).tolist())
    r = dict(seed=seed, tv=tv, light=light, queries=qres, sec=time.time() - t0); res["rows"].append(r)
    print(f"seed {seed}: 表との差 {tv:.3f}  {r['sec']:.0f} 秒")
    for name, x in qres.items():
        print(f"   「{name}」 光の器: 果物 {100 * x['score'][0]:.0f}%・人 {100 * x['score'][1]:.0f}%   表: 果物 {100 * x['truth'][0]:.0f}%・人 {100 * x['truth'][1]:.0f}%")
res.update(OBJ=OBJ, VAL=VAL, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
