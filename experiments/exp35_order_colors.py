"""実験35: 組み合わせ（語順）— 直前 2 文字を別々の色で入れ、カー効果で混ぜると、3-gram（2 文字の組）に届くか

着想: 実験29・30 で、直前の文字ごとの予想を合わせる（重ねる・掛ける）だけでは 2-gram 止まりだった。3-gram が強いのは「この 2 文字の組のあと」を見るから。
      実験14 で、色を分けた 2 つの入力をカー効果（色をまたいで効く）で混ぜると、表計算では解けない「どちらか片方だけ」を 100% 解けた。同じ形で語順の組み合わせを試す。
方法: 青空文庫（作品 0〜299 で学習、300〜349 で試験）、よく出る文字 30 個 + 〈他〉= 記号 31 個（周波数 −30, −28, …、2 おき）、縦 N = 64、列 100、列の間 64 画素。
      色 1 = 直前の文字の波、色 2 = 2 つ前の文字の波（同じ格子・同じ回折を別々に通す）。出口で 2 色の明るさを足して、記号ごとの明るさを次の文字の確率とする。
      曲がる処理: なし（2 色が混ざらない = 2 つの予想の足し算）/ カー効果 u·e^(iγ·N·I)（I = 2 色の明るさの合計、色をまたいで効く）γ = 0.3・1・3。
      格子は周波数 1〜16（縦横とも sin と cos）。学習: よく出る 2 文字の組 600 個の 3-gram の分布（2-gram で補ったもの）を答えにする。Adam（0.01）800 回、seed 0。
比較相手: 数えた 1-gram・2-gram・3-gram（同じ 31 記号）。
指標: 試験の文字の当たり、交差エントロピー（ビット/文字）。
結果の置き場: results/exp35.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp35_order_colors.py
"""
import argparse, collections, glob, json, math, os, re, sys, time, zipfile

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--V", type=int, default=30)
ap.add_argument("--N", type=int, default=64)
ap.add_argument("--cols", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=800)
ap.add_argument("--ctx", type=int, default=600)
ap.add_argument("--kinds", default="なし,カー効果 γ=0.3,カー効果 γ=1,カー効果 γ=3")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp35.txt")
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
top = [c for c, _ in collections.Counter(train).most_common(a.V)]; idx = {c: i for i, c in enumerate(top)}; K = a.V + 1
enc = lambda s: np.array([idx.get(c, a.V) for c in s]); tr, te = enc(train), enc(test)
uni = np.bincount(tr, minlength=K) + 1.0; uni /= uni.sum()
bi = np.ones((K, K)) * 0.01; np.add.at(bi, (tr[:-1], tr[1:]), 1); bi /= bi.sum(1, keepdims=True)
tri = np.zeros((K, K, K)); np.add.at(tri, (tr[:-2], tr[1:-1], tr[2:]), 1)
P3 = (tri + 0.5 * bi[None, :, :]) / (tri.sum(2, keepdims=True) + 0.5)                   # 3-gram（2-gram で補う）
res = dict(exp="35", args=vars(a), rows=[])


def score(name, pr, extra=None):
    y = te[2:]; p = pr[np.arange(len(y)), y]
    r = dict(kind=name, acc=float((pr.argmax(1) == y).mean()), bits=float(-np.log2(np.maximum(p, 1e-12)).mean()), **(extra or {}))
    res["rows"].append(r); print(f"{name:<20} 当たり {100 * r['acc']:5.1f}%  交差エントロピー {r['bits']:.3f}", flush=True)


score("1-gram（数える）", np.tile(uni, (len(te) - 2, 1)))
score("2-gram（数える）", bi[te[1:-1]])
score("3-gram（数える）", P3[te[:-2], te[1:-1]])

N, L = a.N, 2 * a.N; pos = torch.arange(N, dtype=torch.float32); col = torch.arange(a.cols, dtype=torch.float32)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
freq = torch.tensor([2 * (i - K // 2) for i in range(K)]); bins = freq % N
waves = (torch.exp(1j * 2 * math.pi * freq[:, None].float() * pos / N) / math.sqrt(N)).to(torch.complex64)
fr = torch.arange(1, a.F + 1, dtype=torch.float32)
Y1, Y2 = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
X1, X2 = torch.sin(2 * math.pi * fr[:, None] * col / a.cols), torch.cos(2 * math.pi * fr[:, None] * col / a.cols)
ctx = [tuple(c) for c, _ in collections.Counter(zip(tr[:-2].tolist(), tr[1:-1].tolist())).most_common(a.ctx)]
tgt = torch.tensor(np.stack([P3[x, y] for x, y in ctx]), dtype=torch.float32)
mk = lambda cs: torch.stack([torch.stack([waves[y], waves[x]]) for x, y in cs])        # (組, 色 2, N): 色 1 = 直前、色 2 = 2 つ前


def run(A, u, gamma):
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((Y1, X2), (Y1, X1), (Y2, X2), (Y2, X1))))
    u = torch.nn.functional.pad(u, (0, L - N))
    for c in range(a.cols):
        u = torch.cat([u[..., :N] * torch.exp(1j * G[:, c]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
        if gamma:
            u = u * torch.exp(1j * gamma * N * (u.abs() ** 2).sum(1, keepdim=True))     # 2 色の明るさの合計で遅れる（色をまたいで効く）
    W = (torch.fft.fft(u[..., :N], n=N, dim=-1, norm="ortho").abs() ** 2).sum(1)[:, bins]
    return W / W.sum(1, keepdim=True) + 1e-6


for kind in a.kinds.split(","):
    t0 = time.time(); g = torch.Generator().manual_seed(a.seed); gamma = float(kind.split("=")[1]) if "=" in kind else 0.0
    A = torch.nn.Parameter(0.1 * torch.randn(4, a.F, a.F, generator=g)); opt = torch.optim.Adam([A], lr=0.01); u0 = mk(ctx)
    for step in range(a.steps):
        loss = -(tgt * torch.log(run(A, u0, gamma))).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        allc = [(x, y) for x in range(K) for y in range(K)]; T = run(A, mk(allc), gamma).numpy().reshape(K, K, K)
    score(f"光の器・{kind}", T[te[:-2], te[1:-1]], dict(sec=round(time.time() - t0), train_loss=float(loss.detach())))
res.update(sec=time.time() - t00, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
