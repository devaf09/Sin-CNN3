"""実験54: カー効果をどこでかけるか — 場の面 / レンズの焦点面（記号が光の点になる所）。語順の組み合わせ（実験35・38 の続き）

着想: 2026-10-28 持ち主の問い「カー効果での語順（実験35 の失敗）は、非線形のやり方に問題があるってことかな？ 2D のときも平方根で悩んだ。
      同じ解決が難しいのは次元数によるんだろうか」。
仮説: 次元数ではなく、記号の表し方の問題。周波数で表した記号（縦いっぱいの波）は、明るさがどこでもほぼ同じ（1 本の波なら完全に一様）。
      明るさで位相を曲げるカー効果は、どこにも差を付けられず、全体を一様にずらすだけになる。うまくいった実験14 は光の点（場所）で表した入力だった。
      レンズで周波数の面（焦点面）に移すと、周波数の記号は光の点になる。そこでカー効果をかければ、点ごと（記号ごと）に位相が曲がり、
      2 つの記号が同じ点に重なったとき（組み合わせ）だけ特別な曲がり方をするはず。
方法: 実験38 と同じ課題（よく出る 100 文字 + 〈他〉、直前 + 2 つ前の文字、画素ごとの板 20 列 × 段、3-gram の分布 3000 組、1500 回）。段と段の間の非線形を変える:
        1 段（なし）/ 2 段（明るさを測って出し直す、実験38）/
        2 段（カー・場の面）: u ← u·exp(iγ·I/平均 I)、I = |u|²（場の面でそのまま）
        2 段（カー・焦点面）: レンズ（FFT）→ 焦点面で u ← u·exp(iγ·I/平均 I) → レンズ（逆 FFT）→ 2 段目
      カーの強さは明るさの平均で割ってそろえる（実験35 では γ × N × 明るさで、強さの付け方が合っていなかった疑い）。γ = 1・3。
結果の置き場: results/exp54.txt
実行: python experiments/exp54_kerr_focal.py --device cuda（experiments/sweeps/sweep54.sh、GPU の共通の待ち行列）
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
ap.add_argument("--kinds", default="1 段,2 段（明るさ）,2 段（カー・場）γ=1,2 段（カー・場）γ=3,2 段（カー・焦点面）γ=1,2 段（カー・焦点面）γ=3")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp54.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2); t00 = time.time(); torch.set_default_device(a.device)


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
res = dict(exp="54", args=vars(a), rows=[])


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
    for P in Ps[1:]:
        if "カー" in kind:                                                                # カー効果: 明るさで位相を曲げる（明るさの平均でそろえる）
            gam = float(kind.split("γ=")[1])
            if "焦点面" in kind:
                F = torch.fft.fft(u, dim=-1, norm="ortho"); I = F.abs() ** 2
                F = F * torch.exp(1j * gam * I / I.mean(1, keepdim=True)); u = torch.fft.ifft(F, dim=-1, norm="ortho")
            else:
                I = u.abs() ** 2; u = u * torch.exp(1j * gam * I / I.mean(1, keepdim=True))
            u = stage(P, u.to(torch.complex64)); continue
        I = u.abs() ** 2; I = I / I.sum(1, keepdim=True)                                  # 2 段目以降: 明るさを測る（合計 1 にそろえる）
        amp = torch.sqrt(I + 1e-12) if "2 乗" not in kind else I / torch.sqrt((I ** 2).sum(1, keepdim=True))   # 出し直す光の振幅
        u = stage(P, amp.to(torch.complex64))
    W = torch.fft.fft(u, n=N, dim=-1, norm="ortho").abs()[:, bins] ** 2
    return W / W.sum(1, keepdim=True) + 1e-6


ctx = [c for c, _ in collections.Counter(zip(tr[:-2].tolist(), tr[1:-1].tolist())).most_common(a.ctx)]
tgt = torch.tensor(np.stack([P3(x, y) for x, y in ctx]), dtype=torch.float32); u0 = mk(ctx)
for kind in a.kinds.split(","):
    t0 = time.time(); g = torch.Generator().manual_seed(a.seed)
    n_st = int(kind[0]) if kind[0].isdigit() else 1                                     # 「3 段（明るさ）」なら 3 段
    Ps = [torch.nn.Parameter((0.1 * torch.randn(N, a.cols, generator=g, device="cpu")).to(a.device)) for _ in range(n_st)]
    opt = torch.optim.Adam(Ps, lr=0.01)
    for step in range(a.steps):
        loss = -(tgt * torch.log(run(Ps, u0, kind))).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        uniq = sorted(set(zip(te[:-2].tolist(), te[1:-1].tolist()))); out = {}
        for i in range(0, len(uniq), 2000):
            part = uniq[i:i + 2000]; T = run(Ps, mk(part), kind).cpu().numpy(); out.update(zip(part, T))
    score(f"光の器・{kind}", lambda: np.stack([out[(x, y)] for x, y in zip(te[:-2].tolist(), te[1:-1].tolist())]),
          dict(sec=round(time.time() - t0), train_loss=float(loss.detach()), device=a.device))
res.update(sec=time.time() - t00, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
