"""実験30: 前後の観点周波数で次の文字を当てる（持ち主の案）— 直前 9 文字それぞれの「k 文字後ろ」の予想を合わせる

着想: 2026-10-27 持ち主の案「前後に観点周波数を持って、そこの反応率から次の文字を想定する。文字列 N の次の単語は、文字列 N−9 までの観点周波数
      『後ろ』の確率で想定できるのでは。N−1 より N−2 の方が少し確率比率を下げるなどの調整は必要かも」。
方法: 文字・データは実験29 と同じ（青空文庫の作品 0〜299 で学習・数える、300〜349 の最初の 2 万文字で合わせ方の割合を決め、その後の 20 万文字で試す。記号 101 個）。
      観点 k（k = 1〜9）= 縦の周波数を k ずらす。器は「文字 c を観点 k で入れると、c の k 文字後ろに来る文字の分布が光る」ように学習する（数えた表を答えにする）。
      記号は周波数 −500, −490, …（10 おき）、縦 N = 1024。器は実験28 と同じ作り（列 20、列の間 64 画素、曲がる処理なし）、F = 32・64。1500 回、seed 0。
      合わせ方: 掛け算（log P = Σ_k w_k·log P_k(次 | N−k の文字)、w_k = r^(k−1)）/ 足し算（P = Σ_k w_k·P_k / Σ w_k）。r は 0.1〜1 から確認用の文で選ぶ。
      比較相手: 数えた 1-gram・2-gram・3-gram、同じ合わせ方を数えた表でした場合（合わせ方そのものの力）。
指標: 試験の文字の当たり、交差エントロピー（ビット/文字）。
結果の置き場: results/exp30.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp30_offset_views.py
"""
import argparse, collections, glob, json, math, os, re, sys, time, zipfile

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--Fs", default="32,64")
ap.add_argument("--K", type=int, default=9, help="何文字前まで使うか")
ap.add_argument("--N", type=int, default=1024)
ap.add_argument("--cols", type=int, default=20)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp30.txt")
ap.add_argument("--save", default="", help="学習した格子の強さを保存するファイル（例: results/exp30_grid_F32.npz）。単語ごとのマップ（tools/word_maps.py）に使う")
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


train, held = texts(0, 300), texts(300, 350)
top = [c for c, _ in collections.Counter(train).most_common(100)]; idx = {c: i for i, c in enumerate(top)}; S = 101
enc = lambda s: np.array([idx.get(c, 100) for c in s]); tr = enc(train); va, te = enc(held[:20000]), enc(held[20000:220000])
uni = np.bincount(tr, minlength=S) + 1.0; uni /= uni.sum()
T = np.ones((a.K + 1, S, S)) * 0.01                                                   # T[k][c] = c の k 文字後ろの分布（数えた表）
for k in range(1, a.K + 1):
    np.add.at(T[k], (tr[:-k], tr[k:]), 1)
T /= T.sum(2, keepdims=True)
tri = collections.defaultdict(lambda: np.zeros(S))
for x, y, z in zip(tr[:-2], tr[1:-1], tr[2:]):
    tri[(x, y)][z] += 1
res = dict(exp="30", args=vars(a), rows=[])


def combine(tab, seq, r, how):
    """seq の各位置（K 文字目以降）で、直前 K 文字の「k 文字後ろ」の予想 tab[k][文字] を合わせた確率 (位置, S)。"""
    n = len(seq) - a.K; w = r ** np.arange(a.K)
    P = np.stack([tab[k][seq[a.K - k:a.K - k + n]] for k in range(1, a.K + 1)])        # (K, 位置, S)
    if how == "掛け算":
        lp = (w[:, None, None] * np.log(P)).sum(0); lp -= lp.max(1, keepdims=True); p = np.exp(lp)
    else:
        p = (w[:, None, None] * P).sum(0)
    return p / p.sum(1, keepdims=True)


def report(name, p, seq, extra=None):
    y = seq[a.K:]; q = p[np.arange(len(y)), y]
    r = dict(kind=name, acc=float((p.argmax(1) == y).mean()), bits=float(-np.log2(np.maximum(q, 1e-12)).mean()), **(extra or {}))
    res["rows"].append(r); print(f"{name:<34} 当たり {100 * r['acc']:5.1f}%  交差エントロピー {r['bits']:.3f}" + (f"  {extra}" if extra else ""), flush=True)


def best_r(tab, how):
    """確認用の文で、交差エントロピーが一番小さい r を選ぶ。"""
    rs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    bits = [float(-np.log2(combine(tab, va, r, how)[np.arange(len(va) - a.K), va[a.K:]]).mean()) for r in rs]
    return rs[int(np.argmin(bits))]


report("1-gram（数える）", np.tile(uni, (len(te) - a.K, 1)), te)
report("2-gram（数える）", T[1][te[a.K - 1:-1]], te)
P3 = lambda x, y: (tri[(x, y)] + 0.5 * T[1][y]) / (tri[(x, y)].sum() + 0.5)
report("3-gram（数える）", np.stack([P3(x, y) for x, y in zip(te[a.K - 2:-2], te[a.K - 1:-1])]), te)
for how in ("掛け算", "足し算"):
    r = best_r(T, how); report(f"前 {a.K} 文字の観点・{how}（数えた表）", combine(T, te, r, how), te, dict(r=r))

N, L = a.N, 2 * a.N; pos = torch.arange(N, dtype=torch.float32); col = torch.arange(a.cols, dtype=torch.float32)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
freq = torch.tensor([10 * (i - S // 2) for i in range(S)]); bins = freq % N
pairs = [(c, k) for k in range(1, a.K + 1) for c in range(S)]
u0 = torch.stack([torch.exp(1j * 2 * math.pi * (freq[c] + k) * pos / N) / math.sqrt(N) for c, k in pairs]).to(torch.complex64)
tgt = torch.tensor(np.stack([T[k][c] for c, k in pairs]), dtype=torch.float32)
for F in [int(f) for f in a.Fs.split(",")]:
    t0 = time.time(); g = torch.Generator().manual_seed(a.seed); fr = torch.arange(1, F + 1, dtype=torch.float32)
    Y1, Y2 = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
    X1, X2 = torch.sin(2 * math.pi * fr[:, None] * col / a.cols), torch.cos(2 * math.pi * fr[:, None] * col / a.cols)
    A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); opt = torch.optim.Adam([A], lr=0.01)

    def run(u):
        G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((Y1, X2), (Y1, X1), (Y2, X2), (Y2, X1))))
        u = torch.nn.functional.pad(u, (0, L - N))
        for c in range(a.cols):
            u = torch.cat([u[:, :N] * torch.exp(1j * G[:, c]), u[:, N:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
        W = torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs()[:, bins] ** 2
        return W / W.sum(1, keepdim=True) + 1e-6

    for step in range(a.steps):
        loss = -(tgt * torch.log(run(u0))).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        O = run(u0).numpy(); TO = np.ones_like(T)
        for i, (c, k) in enumerate(pairs):
            TO[k][c] = O[i] / O[i].sum()
    tv = float(np.mean([np.abs(TO[k][c] - T[k][c]).sum() / 2 for c, k in pairs]))
    if a.save:
        np.savez_compressed(a.save.replace(".npz", f"_F{F}.npz"), A=A.detach().numpy(), F=F, N=N, cols=a.cols, dz=a.dz, K=a.K, chars="".join(top))
    for how in ("掛け算", "足し算"):
        r = best_r(TO, how); report(f"前 {a.K} 文字の観点・{how}（光の器 F={F}）", combine(TO, te, r, how), te, dict(r=r, tv=round(tv, 3), sec=round(time.time() - t0)))
    report(f"直前 1 文字だけ（光の器 F={F}）", TO[1][te[a.K - 1:-1]], te)
res.update(sec=time.time() - t00, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
