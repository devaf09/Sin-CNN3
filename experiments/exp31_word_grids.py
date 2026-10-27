"""実験31: 単語ごとに格子を 1 枚（格子 = 単語）— 入口は「何か」の光だけ、観点で属性を読み、逆引きで単語を当てる

着想: 2026-10-27 持ち主の案「単語ごとに格子を 1 つ作れば、格子そのものが単語を表すから、入口側に単語を示す周波数を入れなくてもよくなる。
      入力:『何か』↔ 格子（単語）↔ 複数の要素 × 観点周波数、または 入力:『何か』× 観点周波数 A ↔ 格子（単語）↔ 複数の要素 × 観点周波数 A」。
方法: 実験21 の表（物 5・観点 5・値 12、記号の周波数も同じ）を使う。物ごとに小さな格子を 1 枚（周波数 1〜Fw、学習する数 4·Fw²）学習する。
      入口 =「何か」= 一様な光（周波数 0）を観点 A のぶんずらしたもの。器の形は実験21 と同じ（縦 100、100 列、列の間 64 画素、曲がる処理なし）。
        形 1: 出口で値の周波数の明るさを読む。
        形 2: 出口で「値の周波数 + 観点 A のずらし」の明るさを読む（観点が格子を通り抜けて残る）。
      逆引き: 手がかりの値の光（形 2 では観点のずらしも）を、5 枚の格子それぞれに逆向きに通し、入口の「何か × 観点」の明るさを読む。
              格子ごとに値へ届く光の量が違うので、その格子のその観点で値に届いた光の合計で割ってそろえ、掛け算して物の点数にする。
      融合: リンゴの格子とトマトの格子の位相を平均した 1 枚 /  2 枚に光を分けて通して足す、の出口の分布が、2 つの平均にどれだけ近いか。
比較相手: 実験21（共有の格子 1 枚、学習する数 1024、表との差 0.13）、表から計算した点数。1500 回、3 seed。
結果の置き場: results/exp31.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp31_word_grids.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--Fws", default="4,8")
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp31.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp21_viewpoint.py"), encoding="utf-8").read().split("res = dict(")[0]
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)                   # 実験21 の表・記号・回折をそのまま使う
OBJ, VAL, VIEW, TABLE, QUERIES, freq, N, L, H = (ns[k] for k in ("OBJ", "VAL", "VIEW", "TABLE", "QUERIES", "freq", "N", "L", "H"))
pos = torch.arange(N, dtype=torch.float32)
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)
vbins = torch.tensor([freq[v] for v in VAL])


def basis(Fw):
    fr = torch.arange(1, Fw + 1, dtype=torch.float32)
    return torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)


def grid(A, B):
    sy, cy = B
    return sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))


def prop(G, u, back=False):
    u = torch.nn.functional.pad(u, (0, L - N))
    for col in (reversed(range(N)) if back else range(N)):
        if back:
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * torch.conj(H), n=L, dim=-1, norm="ortho")
            u = torch.cat([u[:, :N] * torch.exp(-1j * G[:, col]), u[:, N:]], 1)
        else:
            u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs() ** 2


views = list(VIEW); d = torch.tensor([VIEW[v] for v in views])
probe = torch.stack([wave(VIEW[v]) for v in views])                                       # 「何か」× 観点
res = dict(exp="31", args=vars(a), rows=[])
for form in (1, 2):
    for Fw in [int(f) for f in a.Fws.split(",")]:
        for seed in range(a.seeds):
            t0 = time.time(); g = torch.Generator().manual_seed(seed); B = basis(Fw); Gs = {}; tvs = []; lit = {}
            read = lambda P: P[:, (vbins[None, :] + (d[:, None] if form == 2 else 0)) % N].diagonal(dim1=0, dim2=0) if False else torch.stack([P[i, (vbins + (d[i] if form == 2 else 0)) % N] for i in range(len(views))])
            for o in OBJ:
                q = torch.zeros(len(views), len(VAL) + 1)
                for i, v in enumerate(views):
                    for val, p in TABLE[o][v].items():
                        q[i, VAL.index(val)] = p
                A = torch.nn.Parameter(0.1 * torch.randn(4, Fw, Fw, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
                for step in range(a.steps):
                    W = read(prop(grid(A, B), probe)); p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
                    loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    Gs[o] = grid(A, B); W = read(prop(Gs[o], probe)); D = W / W.sum(1, keepdim=True); lit[o] = W.sum(1).numpy()   # 観点ごとに値へ届いた光（格子ごとの明るさの違いをそろえるのに使う）
                    tvs.append(float((D - q[:, :-1]).abs().sum(1).mean() / 2))
            with torch.no_grad():
                back = {}
                for v in views:                                                            # 逆引き: 値の光を 5 枚の格子に逆向きに
                    i = views.index(v)
                    for val in VAL:
                        uL = wave(freq[val] + (VIEW[v] if form == 2 else 0))[None]
                        back[(v, val)] = np.array([float(prop(Gs[o], uL, back=True)[0, VIEW[v] % N]) / lit[o][i] for o in OBJ])   # 格子ごとの明るさでそろえる
                qres = {}
                for name, ev in QUERIES.items():
                    sc = np.prod([back[(v, val)] for v, val in ev.items()], axis=0)
                    tr = np.prod([[TABLE[o][v].get(val, 0) for o in OBJ] for v, val in ev.items()], axis=0)
                    qres[name] = dict(score=(sc / sc.sum()).tolist(), truth=(tr / tr.sum()).tolist(), top=OBJ[int(np.argmax(sc))], top_truth=OBJ[int(np.argmax(tr))])
                Wa, Wt = read(prop(Gs["リンゴ"], probe)), read(prop(Gs["トマト"], probe))           # 融合
                target = (Wa / Wa.sum(1, keepdim=True) + Wt / Wt.sum(1, keepdim=True)) / 2
                Wavg = read(prop((Gs["リンゴ"] + Gs["トマト"]) / 2, probe)); Wsum = Wa + Wt
                fuse = {k: float((W / W.sum(1, keepdim=True) - target).abs().sum(1).mean() / 2) for k, W in (("位相の平均", Wavg), ("2 枚に分けて足す", Wsum))}
            r = dict(form=form, Fw=Fw, params_per_word=4 * Fw * Fw, seed=seed, tv=float(np.mean(tvs)), queries=qres, fuse=fuse, sec=time.time() - t0)
            res["rows"].append(r)
            print(f"形 {form} Fw={Fw}（1 単語 {4 * Fw * Fw} 個 = {16 * Fw * Fw} バイト）seed {seed}: 表との差 {r['tv']:.3f}  逆引き " +
                  " ".join(f"「{k[:6]}」{x['top']}{'○' if x['top'] == x['top_truth'] else '×'}" for k, x in qres.items()) +
                  f"  融合（平均への近さ）位相の平均 {fuse['位相の平均']:.3f}・2 枚で足す {fuse['2 枚に分けて足す']:.3f}  {r['sec']:.0f} 秒", flush=True)
res.update(OBJ=OBJ, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
