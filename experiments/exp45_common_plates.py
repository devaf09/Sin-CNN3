"""実験45: 共通の板（初期の板）の型の比べ — 学習の速さ・拡散しにくさ

着想: 2026-10-28 持ち主「今後も共通の板を使うことが増えると思うので、いくつかパターンを作っておくのもいい。基準の一つが学習の速さ（それ以外では拡散しにくいなど）。
      例えば入側の左側の周波数軸を高めに、右に行くほど低めとか、それも直線的な周波数変化ではなくて反比例な感じにするとか、横軸は完全にホワイトノイズみたいな散乱度にするとか」。
      実験36 で、初期の板の形は学習後も残る（初期が強いと変化は 2 割）と分かった。だから初期の板の型が、覚えやすさや融合のしやすさを決めるはず。
方法: 実験42 と同じ課題（実験21 の表、使う周波数 37 個を低い所に詰める）、画素ごとの板（縦 128、列 20、列の間 64 画素）。初期の板の型:
        平ら（0）/ 白色雑音（強さ 0.1・1・一様 ±π）/ 縦の縞で左ほど周波数が高い（直線・反比例、40 → 2 周）/ 逆に左ほど低い（直線、2 → 40）/
        縦はなめらか・列ごとにばらばら（横は白色雑音）/ レンズ（列ごとに 2 次の位相、光を集める向き）
      縞の型は 位相 = 1.0·sin(2π·f_列·y/128 + でたらめな位相)。Adam（0.01）600 回、3 seed。
指標: 表との差が 0.15 を切った回（学習の速さ、25 回ごとに測る）、600 回後の表との差、値に届いた光（拡散しにくさ。大きいほど光が答えに集まる）、
      学習で板がどれだけ動いたか（|P − P0| / √画素数）。
結果の置き場: results/exp45.txt
実行: OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp45_common_plates.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=128)
ap.add_argument("--cols", type=int, default=20)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp45.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp21_viewpoint.py"), encoding="utf-8").read().split("\nres = dict(exp=")[0]
sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)
OBJ, VAL, VIEW, TABLE = ns["OBJ"], ns["VAL"], ns["VIEW"], ns["TABLE"]
N, L, C = a.N, 2 * a.N, a.cols; y = torch.arange(N, dtype=torch.float32)[:, None]; c = torch.arange(C, dtype=torch.float32)[None, :]
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
packed = {(o, v): 6 * i + VIEW[v] for i, o in enumerate(OBJ) for v in VIEW}; packed.update({x: 30 + j for j, x in enumerate(VAL)})
wave = lambda k: (torch.exp(1j * 2 * math.pi * k * y[:, 0] / N) / math.sqrt(N)).to(torch.complex64)
pairs = [(o, v) for o in OBJ for v in VIEW]; u0 = torch.stack([wave(packed[(o, v)]) for o, v in pairs]); vb = torch.tensor([packed[x] for x in VAL])
q = torch.zeros(len(pairs), len(VAL) + 1)
for k, (o, v) in enumerate(pairs):
    for x, p in TABLE[o][v].items():
        q[k, VAL.index(x)] = p


def stripes(f_of_col, g):
    ph = 2 * math.pi * torch.rand(1, C, generator=g)
    return torch.sin(2 * math.pi * f_of_col * y / N + ph)


def smooth_cols(g):
    """縦はなめらか（周波数 1〜8 の和）、列ごとにばらばら。"""
    fr = torch.arange(1, 9, dtype=torch.float32)[:, None, None]
    amp = torch.randn(8, 1, C, generator=g) / 3; ph = 2 * math.pi * torch.rand(8, 1, C, generator=g)
    return (amp * torch.sin(2 * math.pi * fr * y[None] / N + ph)).sum(0)


lin_dn = 40 + (2 - 40) * c / (C - 1); inv_dn = 40 / (1 + c * (40 / 2 - 1) / (C - 1)); lin_up = 2 + (40 - 2) * c / (C - 1)
PLATES = {"平ら（0）": lambda g: torch.zeros(N, C), "白色雑音 0.1": lambda g: 0.1 * torch.randn(N, C, generator=g),
          "白色雑音 1": lambda g: torch.randn(N, C, generator=g), "白色雑音 ±π": lambda g: math.pi * (2 * torch.rand(N, C, generator=g) - 1),
          "左ほど高い（直線）": lambda g: stripes(lin_dn, g), "左ほど高い（反比例）": lambda g: stripes(inv_dn, g),
          "左ほど低い（直線）": lambda g: stripes(lin_up, g), "縦なめらか・横ばらばら": smooth_cols,
          "レンズ（2 次の位相）": lambda g: -0.002 * (y - N / 2) ** 2 * torch.ones(1, C) + 0.1 * torch.randn(N, C, generator=g)}


def run(P, u):
    u = torch.nn.functional.pad(u, (0, L - N))
    for k in range(C):
        u = torch.cat([u[:, :N] * torch.exp(1j * P[:, k]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs()[:, vb] ** 2


res = dict(exp="45", args=vars(a), rows=[])
for name, mk in PLATES.items():
    for seed in range(a.seeds):
        t0 = time.time(); g = torch.Generator().manual_seed(seed); P0 = mk(g).float(); P = torch.nn.Parameter(P0.clone()); opt = torch.optim.Adam([P], lr=0.01); hit = None
        for step in range(a.steps):
            W = run(P, u0); p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
            loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if hit is None and step % 25 == 0 and float(((W / W.sum(1, keepdim=True)) - q[:, :-1]).abs().sum(1).mean() / 2) < 0.15:
                hit = step
        with torch.no_grad():
            W = run(P, u0); tv = float(((W / W.sum(1, keepdim=True)) - q[:, :-1]).abs().sum(1).mean() / 2)
            r = dict(plate=name, seed=seed, steps_to_015=hit if hit is not None else a.steps, tv=tv, light=float(W.sum(1).mean()),
                     move=float((P - P0).norm() / math.sqrt(P0.numel())), sec=time.time() - t0)
        res["rows"].append(r)
        print(f"{name:<14} seed {seed}: 0.15 を切った回 {r['steps_to_015']:>4}  表との差 {tv:.3f}  値に届いた光 {100 * r['light']:4.1f}%  板の動き {r['move']:.2f}  {r['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
