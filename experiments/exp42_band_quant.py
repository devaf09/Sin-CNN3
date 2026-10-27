"""実験42・43: 周波数の高低の影響と、観点の波の量子化（コンパクト化）

着想: 2026-10-28 持ち主の依頼「観点周波数の層の限界と、それら以外の周波数の高低がどう影響するのかも知っておきたい」
      「観点周波数そのものも量子化のようなコンパクト化が可能か実験で試してほしい」。
方法: 実験21 の表（物 5・観点 5・値 12、使う周波数 37 個）。器は画素ごとの板（縦 256、列 20、列の間 64 画素）。1500 回、3 seed。
      実験42（高低）: 37 個を詰めて（実験33）置く場所を変える。
        低い: 周波数 0〜41 / 中くらい: 60〜101 / 高い: 86〜127（上限 128 = 場の幅の半分の近く）/ 散らす: 詰めた番号 × 97 mod 256（全体に散らす）
      実験43（量子化）: 置き場所は「低い」。入口・逆引きで使う波の位相を b ビット（2^b 段）に丸める（安い板で作る）。b = 1・2・3・丸めない。
        1 ビット = 位相 0 か π（白黒の縞）。丸めると、もとの周波数のほかに高調波（3 倍・5 倍…の周波数）が出る。
指標: 順引きの表との差、問い合わせ（実験21 の 4 問）が表の正解と同じ物を一番にした数、値に届いた光。
結果の置き場: results/exp42.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp42_band_quant.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=256)
ap.add_argument("--cols", type=int, default=20)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp42.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp21_viewpoint.py"), encoding="utf-8").read().split("\nres = dict(exp=")[0]
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)
OBJ, VAL, VIEW, TABLE, QUERIES = ns["OBJ"], ns["VAL"], ns["VIEW"], ns["TABLE"], ns["QUERIES"]
N, L = a.N, 2 * a.N; pos = torch.arange(N, dtype=torch.float32)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
packed = {(o, v): 6 * i + VIEW[v] for i, o in enumerate(OBJ) for v in VIEW}; packed.update({x: 30 + j for j, x in enumerate(VAL)})   # 0〜41 に詰める
PLACE = {"低い（0〜41）": lambda j: j, "中くらい（60〜101）": lambda j: j + 60, "高い（86〜127）": lambda j: j + 86, "散らす（× 97）": lambda j: (j * 97) % N}


def wave(k, bits):
    ph = 2 * math.pi * k * pos / N
    if bits:
        lv = 2 ** bits; ph = torch.round(ph / (2 * math.pi) * lv) * (2 * math.pi / lv)   # 位相を 2^bits 段に丸める
    return (torch.exp(1j * ph) / math.sqrt(N)).to(torch.complex64)


def prop(P, u, back=False):
    u = torch.nn.functional.pad(u, (0, L - N))
    for c in (reversed(range(a.cols)) if back else range(a.cols)):
        if back:
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * torch.conj(H), n=L, dim=-1, norm="ortho")
            u = torch.cat([u[:, :N] * torch.exp(-1j * P[:, c]), u[:, N:]], 1)
        else:
            u = torch.cat([u[:, :N] * torch.exp(1j * P[:, c]), u[:, N:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs() ** 2


pairs = [(o, v) for o in OBJ for v in VIEW]
q = torch.zeros(len(pairs), len(VAL) + 1)
for k, (o, v) in enumerate(pairs):
    for x, p in TABLE[o][v].items():
        q[k, VAL.index(x)] = p
cases = [("実験42", pl, 0) for pl in PLACE] + [("実験43", "低い（0〜41）", b) for b in (3, 2, 1)]
res = dict(exp="42-43", args=vars(a), rows=[])
for part, pl, bits in cases:
    f = lambda key: PLACE[pl](packed[key]); vb = torch.tensor([f(x) for x in VAL])
    u0 = torch.stack([wave(f((o, v)), bits) for o, v in pairs])
    for seed in range(a.seeds):
        t0 = time.time(); g = torch.Generator().manual_seed(seed)
        P = torch.nn.Parameter(0.1 * torch.randn(N, a.cols, generator=g)); opt = torch.optim.Adam([P], lr=0.01)
        for step in range(a.steps):
            W = prop(P, u0)[:, vb]; p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
            loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            W = prop(P, u0)[:, vb]; D = W / W.sum(1, keepdim=True); tv = float((D - q[:, :-1]).abs().sum(1).mean() / 2); light = float(W.sum(1).mean())
            back = {(v, x): np.array([float(prop(P, wave(f(x), bits)[None], back=True)[0, f((o, v))]) for o in OBJ]) for v in VIEW for x in VAL}
            hits = sum(int(np.argmax(np.prod([back[(v, x)] for v, x in ev.items()], axis=0)) ==
                           np.argmax(np.prod([[TABLE[o][v].get(x, 0) for o in OBJ] for v, x in ev.items()], axis=0))) for ev in QUERIES.values())
        r = dict(part=part, place=pl, bits=bits, seed=seed, tv=tv, light=light, hits=hits, sec=time.time() - t0); res["rows"].append(r)
        print(f"{part} {pl} 位相 {bits or '丸めない'} ビット seed {seed}: 表との差 {tv:.3f}  値に届いた光 {100 * light:.0f}%  問い合わせ {hits}/4  {r['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
