"""実験33: Sin-CNN2 の実験35（間引き = 周波数の並べ直し）を、観点周波数の小さな器に当てはめる

着想: 2026-10-27 持ち主の依頼「以前作った安定した小さなモデルで良いので、Sin-CNN2 の実験35 を同じように適応できるかやってみて」。
      Sin-CNN2 の実験35: 文字の周波数 f を f·g mod L に並べ直す（g は L と互いに素な奇数、約 0.618L）と、小さな L でも周波数が重ならず、
      L を 1/4 にしても正確さは同じかむしろ良く、計算は 1/2〜1/4 になった（低い周波数しか測れないオシロスコープでも、間隔を工夫すれば高い周波数を測れる、という持ち主の案）。
      必要な L は「区別したい（記号, 位置）の数」で決まる。
方法: 安定した小さな器 = 実験21（物 5・観点 5・値 12、リンゴとトマト）。使う周波数は「物 + 観点のずらし」25 個と「値」12 個の計 37 個。
      従来の並べ方（実験21）は周波数 −48〜53 に置くので、縦 N = 100 が要る。3 つの置き方を比べる:
        そのまま: 元の周波数を mod N（N が小さいと重なる）
        詰める: 使う 37 個だけを番号 0〜41 に隙間なく置く（物 5 つは 0, 6, 12, 18, 24 に置き、観点のずらし 1〜5 がその間に入る。値は 30〜41）
        詰めて並べ直す（Sin-CNN2 の実験35）: 詰めた番号 j を (j·g) mod N に置く（g は N と互いに素で約 0.618N）。重ならないまま、隣どうしを遠くに散らす
      ※ 掛け算 g は mod N で 1 対 1 なので、「そのまま」に g を掛けても重なりは減らない（重なるかどうかは mod N で決まる）。重なりを減らすのは「詰める」。
      N = 100・64・48・42 で比べる。光が通る列は 100 で固定（縦の幅だけの効果を見る）。
      「重なり」= 使う 37 個の周波数のうち、ほかと同じ番号になったものの割合。器・学習・逆引きは実験21 と同じ（1500 回、3 seed）。
指標: 順引きの表との差、問い合わせ（実験21 の 4 問）が表の正解と同じ物を一番にした数、1 回の学習の時間。
結果の置き場: results/exp33.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp33_renumber.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--Ns", default="100,64,48,42")
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--cols", type=int, default=100)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp33.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp21_viewpoint.py"), encoding="utf-8").read().split("res = dict(")[0]
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)
OBJ, VAL, VIEW, TABLE, QUERIES, freq0 = (ns[k] for k in ("OBJ", "VAL", "VIEW", "TABLE", "QUERIES", "freq"))
pairs = [(o, v) for o in OBJ for v in VIEW]
used = [freq0[o] + VIEW[v] for o, v in pairs] + [freq0[x] for x in VAL]                   # 使う周波数 37 個（元の番号）
res = dict(exp="33", args=vars(a), rows=[])


def coprime_g(N):
    g = int(round(0.618 * N)) | 1
    while math.gcd(g, N) != 1:
        g += 2
    return g


for N in [int(n) for n in a.Ns.split(",")]:
    for mode in ("そのまま", "詰める", "詰めて並べ直す"):
        g = coprime_g(N) if mode == "詰めて並べ直す" else 1
        packed = {freq0[o] + VIEW[v]: 6 * i + VIEW[v] for i, o in enumerate(OBJ) for v in VIEW}
        packed.update({freq0[x]: 30 + j for j, x in enumerate(VAL)})                       # 使う 37 個を 0〜41 に詰める
        f = (lambda x: x % N) if mode == "そのまま" else (lambda x, g=g: (packed[x] * g) % N)
        slots = [f(x) for x in used]; overlap = 1 - len(set(slots)) / len(slots)
        L = 2 * N; pos = torch.arange(N, dtype=torch.float32); col = torch.arange(a.cols, dtype=torch.float32)
        H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
        fr = torch.arange(1, a.F + 1, dtype=torch.float32)
        Y1, Y2 = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
        X1, X2 = torch.sin(2 * math.pi * fr[:, None] * col / a.cols), torch.cos(2 * math.pi * fr[:, None] * col / a.cols)
        wave = lambda k: (torch.exp(1j * 2 * math.pi * k * pos / N) / math.sqrt(N)).to(torch.complex64)
        vb = torch.tensor([f(freq0[x]) for x in VAL])

        def prop(G, u, back=False):
            u = torch.nn.functional.pad(u, (0, L - N))
            for c in (reversed(range(a.cols)) if back else range(a.cols)):
                if back:
                    u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * torch.conj(H), n=L, dim=-1, norm="ortho")
                    u = torch.cat([u[:, :N] * torch.exp(-1j * G[:, c]), u[:, N:]], 1)
                else:
                    u = torch.cat([u[:, :N] * torch.exp(1j * G[:, c]), u[:, N:]], 1)
                    u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
            return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs() ** 2

        u0 = torch.stack([wave(f(freq0[o] + VIEW[v])) for o, v in pairs])
        q = torch.zeros(len(pairs), len(VAL) + 1)
        for k, (o, v) in enumerate(pairs):
            for x, p in TABLE[o][v].items():
                q[k, VAL.index(x)] = p
        for seed in range(a.seeds):
            t0 = time.time(); gen = torch.Generator().manual_seed(seed)
            A = torch.nn.Parameter(0.1 * torch.randn(4, a.F, a.F, generator=gen)); opt = torch.optim.Adam([A], lr=0.01)
            grid = lambda: sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((Y1, X2), (Y1, X1), (Y2, X2), (Y2, X1))))
            for step in range(a.steps):
                W = prop(grid(), u0)[:, vb]; p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
                loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            sec = time.time() - t0
            with torch.no_grad():
                G = grid(); W = prop(G, u0)[:, vb]; D = W / W.sum(1, keepdim=True); tv = float((D - q[:, :-1]).abs().sum(1).mean() / 2)
                back = {(v, x): np.array([float(prop(G, wave(f(freq0[x]))[None], back=True)[0, f(freq0[o] + VIEW[v])]) for o in OBJ]) for v in VIEW for x in VAL}
                hits = 0
                for name, ev in QUERIES.items():
                    sc = np.prod([back[(v, x)] for v, x in ev.items()], axis=0)
                    tr = np.prod([[TABLE[o][v].get(x, 0) for o in OBJ] for v, x in ev.items()], axis=0)
                    hits += int(np.argmax(sc) == np.argmax(tr))
            r = dict(N=N, mode=mode, g=g, overlap=overlap, seed=seed, tv=tv, hits=hits, sec=sec); res["rows"].append(r)
            print(f"N={N:<3} {mode:<5}（g={g}）重なり {100 * overlap:4.1f}%  seed {seed}: 表との差 {tv:.3f}  問い合わせ {hits}/4  {sec:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
