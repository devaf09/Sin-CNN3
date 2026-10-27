"""実験38: 組み合わせ（語順）を 2 段の器で — 1 段目の出口の明るさを測って出し直し、2 段目へ（画素ごとの板）

着想: 実験29・30・35 で、直前の文字ごとの予想を合わせるだけでは 2-gram 止まり、色を分けたカー効果は学習が崩れた。
      明るさを測る（2 乗）と、同じ色の 2 つの光の干渉の項（振幅の掛け算 = 2 つの文字の組み合わせ）が自然に出る（実験12）。
      それを光として出し直して 2 段目の器に通せば、「この 2 文字の組のあと」を表せるはず。前のプロジェクトの「測って出し直す」非線形（平方根）、
      実験22 の 2 乗の部品と同じ光の部品。
方法: 記号は実験29 と同じ（よく出る 100 文字 + 〈他〉、周波数 2 おき、縦 256、列 20、列の間 64 画素）。入口 = 直前の文字の波 + 2 つ前の文字の波（縦の周波数を 1 ずらす）。
      板は画素ごとに自由な位相（実験28 で規模に強かった形）。3 とおり:
        1 段: 板 20 列 → 出口の記号の明るさ（曲がる処理なし。実験29 の器 2 と同じ形）
        2 段（明るさ）: 板 20 列 → 明るさ I を測る → √I の光を出し直す → 板 20 列 → 出口
        2 段（明るさの 2 乗）: 同じで、出し直す光を I にする（明るさの差を広げる）
      よく出る 2 文字の組 3000 個の 3-gram で学習（Adam 0.01、1500 回）、seed 0。
比較相手: 数えた 1-gram・2-gram・3-gram、実験29 の光の器 1（23.8%）・器 2（21.7%）。
結果の置き場: results/exp38.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp38_two_stage_order.py
"""
import argparse, collections, glob, json, math, os, re, sys, time, zipfile

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--cols", type=int, default=20)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--ctx", type=int, default=3000)
ap.add_argument("--kinds", default="1 段,2 段（明るさ）,2 段（明るさの 2 乗）")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp38.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2); t00 = time.time()


def texts(lo, hi):
    out = []
    for f in sorted(glob.glob("D:/AI_git/Project_Kotoyui/data/aozora/raw/*.zip"))[lo:hi]:
        try:
            with zipfile.ZipFile(f) as z:
                for name in z.namelist():
                    if name.endswith(".txt"):
                        out.append(re.sub(r"《[^》]*》|［＃[^］]*］|｜|\s", "", z.read(name).decode("cp932", errors="ignore")))
        except zipfile.BadZipFile:
            pass
    return "".join(out)


train, test = texts(0, 300), texts(300, 350)[:200000]
V = 100; top = [c for c, _ in collections.Counter(train).most_common(V)]; idx = {c: i for i, c in enumerate(top)}; K = V + 1
enc = lambda s: np.array([idx.get(c, V) for c in s]); tr, te = enc(train), enc(test)
uni = np.bincount(tr, minlength=K) + 1.0; uni /= uni.sum()
bi = np.ones((K, K)) * 0.01; np.add.at(bi, (tr[:-1], tr[1:]), 1); bi /= bi.sum(1, keepdims=True)
tri = collections.defaultdict(lambda: np.zeros(K))
for x, y, z in zip(tr[:-2], tr[1:-1], tr[2:]):
    tri[(x, y)][z] += 1
P3 = lambda x, y: (tri[(x, y)] + 0.5 * bi[y]) / (tri[(x, y)].sum() + 0.5) if (x, y) in tri else bi[y]
res = dict(exp="38", args=vars(a), rows=[])


def score(name, prob_fn, extra=None):
    pr = prob_fn(); y = te[2:]; p = pr[np.arange(len(y)), y]
    r = dict(kind=name, acc=float((pr.argmax(1) == y).mean()), bits=float(-np.log2(np.maximum(p, 1e-12)).mean()), **(extra or {}))
    res["rows"].append(r); print(f"{name:<22} 当たり {100 * r['acc']:5.1f}%  交差エントロピー {r['bits']:.3f}", flush=True)


score("1-gram（数える）", lambda: np.tile(uni, (len(te) - 2, 1)))
score("2-gram（数える）", lambda: bi[te[1:-1]])
score("3-gram（数える）", lambda: np.stack([P3(x, y) for x, y in zip(te[:-2], te[1:-1])]))

N, L = a.N, 2 * a.N; pos = torch.arange(N, dtype=torch.float32)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
freq = torch.tensor([2 * (i - K // 2) for i in range(K)]); bins = freq % N
waves = (torch.exp(1j * 2 * math.pi * freq[:, None].float() * pos / N) / math.sqrt(N)).to(torch.complex64)
shift = torch.exp(1j * 2 * math.pi * pos / N).to(torch.complex64)
mk = lambda cs: torch.stack([waves[y] + waves[x] * shift for x, y in cs]) / math.sqrt(2)


def stage(P, u):
    u = torch.nn.functional.pad(u, (0, L - N))
    for c in range(a.cols):
        u = torch.cat([u[:, :N] * torch.exp(1j * P[:, c]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return u[:, :N]


def run(Ps, u, kind):
    u = stage(Ps[0], u)
    for P in Ps[1:]:                                                                    # 2 段目以降: 明るさを測って出し直す
        I = u.abs() ** 2; I = I / I.sum(1, keepdim=True)                                  # 明るさを測る（合計 1 にそろえる）
        amp = torch.sqrt(I + 1e-12) if "2 乗" not in kind else I / torch.sqrt((I ** 2).sum(1, keepdim=True))   # 出し直す光の振幅
        u = stage(P, amp.to(torch.complex64))
    W = torch.fft.fft(u, n=N, dim=-1, norm="ortho").abs()[:, bins] ** 2
    return W / W.sum(1, keepdim=True) + 1e-6


ctx = [c for c, _ in collections.Counter(zip(tr[:-2].tolist(), tr[1:-1].tolist())).most_common(a.ctx)]
tgt = torch.tensor(np.stack([P3(x, y) for x, y in ctx]), dtype=torch.float32); u0 = mk(ctx)
for kind in a.kinds.split(","):
    t0 = time.time(); g = torch.Generator().manual_seed(a.seed)
    n_st = int(kind[0]) if kind[0].isdigit() else 1                                     # 「3 段（明るさ）」なら 3 段
    Ps = [torch.nn.Parameter(0.1 * torch.randn(N, a.cols, generator=g)) for _ in range(n_st)]
    opt = torch.optim.Adam(Ps, lr=0.01)
    for step in range(a.steps):
        loss = -(tgt * torch.log(run(Ps, u0, kind))).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        uniq = sorted(set(zip(te[:-2].tolist(), te[1:-1].tolist()))); out = {}
        for i in range(0, len(uniq), 2000):
            part = uniq[i:i + 2000]; T = run(Ps, mk(part), kind).numpy(); out.update(zip(part, T))
    score(f"光の器・{kind}", lambda: np.stack([out[(x, y)] for x, y in zip(te[:-2].tolist(), te[1:-1].tolist())]),
          dict(sec=round(time.time() - t0), train_loss=float(loss.detach())))
res.update(sec=time.time() - t00, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
