"""実験29（大規模実験 B-3）: 足りないものを測る — 前の文字から次の文字を当てる（語の順番）

着想: 2026-10-27 持ち主の依頼（大規模な調査と実験、チャットボットまでに何が足りないか）。調査（docs/prior_art.md）の見立てでは、
      今の器は「語の順番」を持たない（観点ごとの掛け算は順番を見ない）。次の文字を当てる課題で、どれだけ足りないかを測る。
方法: 青空文庫（Project_Kotoyui が取得、著作権切れ）の作品 0〜299 で数え、300〜349 で試す。よく出る文字 100 個 + 〈他〉= 記号 101 個（周波数 −100, −98, …、2 おき）。
      比較相手（数えるだけ）: 1-gram（文字の多さ）、2-gram（直前 1 文字）、3-gram（直前 2 文字）。
      光の器（実験28 と同じ作り、縦 256、列 20、列の間 64 画素、周波数 F = 32、曲がる処理なし / カー効果）:
        器 1: 直前 1 文字の波を入れ、出口の記号の明るさを次の文字の確率とする（2-gram を光で）。
        器 2: 直前 1 文字の波 + 直前 2 文字目の波（縦の周波数を 1 ずらす = 位置の観点）を足して入れる。よく出る 2 文字の組 3000 個の 3-gram で学習。
             曲がる処理なしでは、出口は「1 文字目の予想 + 2 文字目の予想」の重ね合わせになり、2 つの文字の組み合わせ（3-gram）は表せないはず。
        器 2 + カー効果: 列ごとに u·e^(iγ|u|²·N)（γ = 1・4。明るさは 1 画素あたりの平均が 1 になるようにそろえて掛ける）。
指標: 試験の文字の当たり（一番明るい = 正解）、交差エントロピー（ビット/文字、小さいほど良い）。seed 0。
結果の置き場: results/exp29.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp29_next_char.py
"""
import argparse, collections, glob, json, math, os, re, sys, time, zipfile

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--F", type=int, default=32)
ap.add_argument("--cols", type=int, default=20)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--ctx", type=int, default=3000, help="器 2 の学習に使う、よく出る 2 文字の組の数")
ap.add_argument("--test_chars", type=int, default=200000)
ap.add_argument("--kinds", default="器 1,器 2,器 2 + カー効果 γ=1,器 2 + カー効果 γ=4")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp29.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2); t00 = time.time()


def texts(lo, hi):
    out = []
    for f in sorted(glob.glob("D:/AI_git/Project_Kotoyui/data/aozora/raw/*.zip"))[lo:hi]:
        try:
            with zipfile.ZipFile(f) as z:
                for name in z.namelist():
                    if name.endswith(".txt"):
                        t = re.sub(r"《[^》]*》|［＃[^］]*］|｜|\s", "", z.read(name).decode("cp932", errors="ignore")); out.append(t)
        except zipfile.BadZipFile:
            pass
    return "".join(out)


train, test = texts(0, 300), texts(300, 350)[:a.test_chars]
V = 100; top = [c for c, _ in collections.Counter(train).most_common(V)]; idx = {c: i for i, c in enumerate(top)}
enc = lambda s: np.array([idx.get(c, V) for c in s]); tr, te = enc(train), enc(test); K = V + 1
uni = np.bincount(tr, minlength=K) + 1.0; uni /= uni.sum()
bi = np.ones((K, K)) * 0.01; np.add.at(bi, (tr[:-1], tr[1:]), 1); bi /= bi.sum(1, keepdims=True)
tri_c = collections.Counter(zip(tr[:-2], tr[1:-1])); tri = collections.defaultdict(lambda: np.zeros(K))
for x, y, z in zip(tr[:-2], tr[1:-1], tr[2:]):
    tri[(x, y)][z] += 1
P2 = lambda x, y: (tri[(x, y)] + 0.5 * bi[y]) / (tri[(x, y)].sum() + 0.5) if (x, y) in tri else bi[y]   # 3-gram（2-gram で補う）
res = dict(exp="29", args=vars(a), chars=dict(train=len(tr), test=len(te)), rows=[])


def score(name, prob_fn):
    pr = prob_fn(); p = pr[np.arange(len(te) - 2), te[2:]]
    r = dict(kind=name, acc=float((pr.argmax(1) == te[2:]).mean()), bits=float(-np.log2(np.maximum(p, 1e-12)).mean()))
    res["rows"].append(r); print(f"{name:<22} 当たり {100 * r['acc']:5.1f}%  交差エントロピー {r['bits']:.3f} ビット/文字", flush=True)


score("1-gram（数える）", lambda: np.tile(uni, (len(te) - 2, 1)))
score("2-gram（数える）", lambda: bi[te[1:-1]])
score("3-gram（数える）", lambda: np.stack([P2(x, y) for x, y in zip(te[:-2], te[1:-1])]))

N, L = a.N, 2 * a.N; dev = a.device
pos = torch.arange(N, dtype=torch.float32); col = torch.arange(a.cols, dtype=torch.float32)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
freq = torch.tensor([2 * (i - K // 2) for i in range(K)]); bins = freq % N
waves = (torch.exp(1j * 2 * math.pi * freq[:, None].float() * pos / N) / math.sqrt(N)).to(torch.complex64)
shift = torch.exp(1j * 2 * math.pi * 1 * pos / N).to(torch.complex64)                # 位置の観点: 2 文字目は縦の周波数を 1 ずらす
fr = torch.arange(1, a.F + 1, dtype=torch.float32)
Y1, Y2 = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
X1, X2 = torch.sin(2 * math.pi * fr[:, None] * col / a.cols), torch.cos(2 * math.pi * fr[:, None] * col / a.cols)


def run(A, u, gamma):
    G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((Y1, X2), (Y1, X1), (Y2, X2), (Y2, X1))))
    u = torch.nn.functional.pad(u, (0, L - N))
    for c in range(a.cols):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, c]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
        if gamma:
            u = u * torch.exp(1j * gamma * N * u.abs() ** 2)
    W = torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs()[:, bins] ** 2
    return W / W.sum(1, keepdim=True) + 1e-6


for kind in a.kinds.split(","):
    t0 = time.time(); g = torch.Generator().manual_seed(a.seed)
    gamma = float(kind.split("=")[1]) if "カー" in kind else 0.0
    if kind == "器 1":
        ctx = [(None, y) for y in range(K)]; tgt = torch.tensor(bi, dtype=torch.float32)
    else:
        ctx = [c for c, _ in tri_c.most_common(a.ctx)]; tgt = torch.tensor(np.stack([P2(x, y) for x, y in ctx]), dtype=torch.float32)
    mk = lambda cs: torch.stack([waves[y] + (waves[x] * shift if x is not None else 0) for x, y in cs]) / (math.sqrt(2) if cs[0][0] is not None else 1)
    u0 = mk(ctx); A = torch.nn.Parameter(0.1 * torch.randn(4, a.F, a.F, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
    for step in range(a.steps):
        loss = -(tgt * torch.log(run(A, u0, gamma))).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        if kind == "器 1":
            T = run(A, u0, gamma).numpy(); fn = lambda: T[te[1:-1]]
        else:
            uniq = sorted(set(zip(te[:-2].tolist(), te[1:-1].tolist()))); out = {}
            for i in range(0, len(uniq), 2000):
                part = uniq[i:i + 2000]; T = run(A, mk(part), gamma).numpy(); out.update(zip(part, T))
            fn = lambda: np.stack([out[(x, y)] for x, y in zip(te[:-2].tolist(), te[1:-1].tolist())])
    score(f"光の{kind}", fn); res["rows"][-1]["sec"] = time.time() - t0
res.update(sec=time.time() - t00, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
