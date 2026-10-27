"""実験25: 文の例から属性の割合を学ぶ — 青空文庫の「形容詞 + 物の名前」を数えて、観点周波数の器に覚えさせる

着想: 2026-10-27 持ち主の案「光点は周囲の文脈や映像などからそれを学習する。例: 色の周波数選択: 文章から色は赤 80%、緑 20% みたいな」。
      文章のファイルは他のプロジェクトから流用してよい（持ち主）。
方法: tools/aozora_pairs.py で、青空文庫（Project_Kotoyui が取得、著作権の切れた作品）から、物の名前（猫・犬・船・花・雪・月・林檎・蜜柑・空・海・血・火）の
      すぐ前の形容詞を数えた（results/aozora_pairs.json）。観点 4 つ（色・大きさ・温度・味）の割合 = 数えた数の割合。
      数えた数が 3 未満の観点（たとえば猫の温度・味）は「拡散」（値に均等。実験23 の「分からない」）とする。
      器・順引きの学習・逆引き・掛け算は実験23 と同じ（縦 128 画素、記号 25 個 = 周波数 −60, −55, …, 60、観点 = ずらし 1〜4）。1500 回、3 seed。
      問い合わせ: 「白くて冷たい」「赤くて熱い」など。比較相手: 数えた割合から計算した点数（デジタル）。
結果の置き場: results/exp25.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp25_aozora_attr.py
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
ap.add_argument("--out", default="results/exp25.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
C = json.load(open("results/aozora_pairs.json", encoding="utf-8"))                  # tools/aozora_pairs.py で数えたもの
OBJ = list(C["count"]); VALS_OF = C["VIEW"]
VAL = [x for v in VALS_OF.values() for x in v]
SYM = OBJ + VAL; freq = {s: -60 + 5 * i for i, s in enumerate(SYM)}
VIEW = {v: i + 1 for i, v in enumerate(VALS_OF)}                                       # 色 1・大きさ 2・温度 3・味 4
ABSENT = None
TABLE = {o: {v: ({k: n / sum(x.values()) for k, n in x.items() if n} if sum(x.values()) >= 3 else ABSENT) for v, x in C["count"][o].items()} for o in OBJ}
QUERIES = {"白くて冷たい": {"色": "白", "温度": "冷たい"}, "赤くて熱い": {"色": "赤", "温度": "熱い"}, "青くて大きい": {"色": "青", "大きさ": "大"},
           "黒くて冷たい": {"色": "黒", "温度": "冷たい"}, "赤くて小さい": {"色": "赤", "大きさ": "小"}, "白い": {"色": "白"}}
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)


def target(o, v, mode):
    t = TABLE[o][v]
    if t is ABSENT:
        return {x: 1 / len(VALS_OF[v]) for x in VALS_OF[v]}
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
res = dict(exp="25", source=C["source"], args=vars(a), rows=[])
for mode in ("拡散",):
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
            ent = {f"{o}の{v}": float(-(D[n] * torch.log(D[n] + 1e-12)).sum()) for n, (o, v) in enumerate(pairs) if TABLE[o][v] is ABSENT}
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
        print(f"{mode} seed {seed}: 表との差 {tv:.3f}  広がり（エントロピー）" + f"拡散した観点 {len(ent)} 個" + f"  {r['sec']:.0f} 秒")
        for name, x in qres.items():
            top3 = lambda sc: " ".join(f"{OBJ[i]}{100 * sc[i]:.0f}" for i in np.argsort(sc)[::-1][:3])
            print(f"   「{name}」 光の器 {top3(x['score'])}  数えた割合 {top3(x['truth'])}")
res.update(OBJ=OBJ, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
