"""実験68: 章板の形の比べ — 1D の共通の器（縦を広げる）/ 2D の共通の器 / 語ごとの 1D の語板。当たりと「逆引きの走査」

着想: 2026-10-29 持ち主「処理や容量削減に役に立つかな？ 今までのグレインでも入射光を変えて縦幅を広げれば同じことができる？
      章板の形成に、今までのグレインと 2D + N の形で、内部の属性を走査するのに向いてるか、大きい実験でなくても手応えを知りたい」。
方法: 会話の名詞 32 語（非公開の材料）、5 観点、値 52、実験58 の決まり（平らにした目標・出口のずらし）。位相の数をそろえて 3 つの形を比べる:
  1D 共通: 縦 256 の 1D の器、板 32 枚（位相 8192）と 64 枚（位相 16384、語ごとの語板と同じ）。語 w・観点 v の入口 = 周波数 5w + v（160 個、どれも直交）。出口 = 周波数 200〜251（入口と重ならない所）。
  2D 共通: 32 × 32 の 2D の器、板 8 枚（位相 8192）。入口 = 傾きの格子（実験67 と同じ）。
  語ごとの 1D の語板: 1 語 = 縦 64 × 板 8 枚（位相 512）× 32 語 = 16384（src/goban_batch.py、実験58）。
  確かめ:
    a. 当たり（確かめ用の交差エントロピー）。
    b. 逆引きの走査: 属性（観点 v の値 x）の出口から光を「逆向き」に 1 回通し（随伴 = 板の位相を逆に、伝わりを逆向きに）、入口の面で
       語 w・観点 v の入口の波と重なる明るさを読むと、章の全部の語の「x らしさ」が 1 回で出る。
       - 逆向きの明るさが、順向きの明るさ（語ごとに光を通した値）と一致するか（器の性質の確かめ）
       - 走査の上位 5 語が、数えた表の上位 5 語とどれだけ重なるか（属性 20 個の平均）
     語ごとの語板では、同じことに 32 回（語の数）光を通す必要がある。
結果の置き場: results/exp68.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp68_chapter_forms.py
"""
import json, math, os, runpy, sys, time

import torch

sys.argv = [sys.argv[0], "--seeds", "0"]
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "exp56_goban_grow.py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train as train_1d, run as run_1d, evaluate

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t00 = time.time()
NTR, NTE, prior = ns["NTR"], ns["NTE"], ns["prior"]


def shift(T, step=9):
    out = T.clone()
    for v in range(T.shape[1]):
        out[:, v, [(r + step * v) % 47 for r in range(47)]] = T[:, v, :47]
    return out


Wn = 32
T, TE = shift(NTR[:Wn]), shift(NTE[:Wn]); pr = prior(shift(NTR))
TG = (T + 5 * pr[None]) / (T.sum(2, keepdim=True) + 5); NV, NX = TG.shape[1], TG.shape[2]
pr2 = lambda W: 0.98 * W / W.sum(-1, keepdim=True).clamp(min=1e-12) + 0.02 / W.shape[-1]


class Device:
    """共通の器（1D: 縦 N、2D: N × N）。入口の波の番号 → 出口の点の明るさ。forward = 順向き、scan = 逆向き 1 回で全部の入口の明るさ。"""

    def __init__(self, dim, N, K, inputs, outputs, dz, seed):
        self.dim, self.N, self.K, self.L = dim, N, K, 2 * N
        f = torch.fft.fftfreq(self.L)
        if dim == 1:
            self.H = torch.exp(1j * dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * f) ** 2))
        else:
            kx, ky = torch.meshgrid(f, f, indexing="ij")
            self.H = torch.exp(1j * dz * torch.sqrt(((2 * math.pi) ** 2 - (2 * math.pi) ** 2 * (kx ** 2 + ky ** 2)).clamp(min=0)))
        g = torch.Generator().manual_seed(seed); shape = (K, N) if dim == 1 else (K, N, N)
        self.P = torch.nn.Parameter(math.pi * (torch.rand(*shape, generator=g) < 0.5).float())
        self.inputs, self.outputs = inputs, outputs                                           # 入口の周波数（1D: 整数、2D: (横, 縦)）と出口の点

    def wave(self, f):
        N = self.N
        if self.dim == 1:
            y = torch.arange(N, dtype=torch.float32); return (torch.exp(2j * math.pi * f * y / N) / math.sqrt(N)).to(torch.complex64)
        yy, xx = torch.meshgrid(torch.arange(N, dtype=torch.float32), torch.arange(N, dtype=torch.float32), indexing="ij")
        return (torch.exp(2j * math.pi * (f[0] * xx + f[1] * yy) / N) / N).to(torch.complex64)

    def _pad(self, u):
        return torch.nn.functional.pad(u, (0, self.N) if self.dim == 1 else (0, self.N, 0, self.N))

    def _crop(self, u):
        return u[..., :self.N] if self.dim == 1 else u[..., :self.N, :self.N]

    def _mask(self, u, ph):
        if self.dim == 1:
            return torch.cat([u[..., :self.N] * torch.exp(1j * ph), u[..., self.N:]], -1)
        top = torch.cat([u[..., :self.N, :self.N] * torch.exp(1j * ph), u[..., :self.N, self.N:]], -1)
        return torch.cat([top, u[..., self.N:, :]], -2)

    def _fft(self, u, inv=False, H=None):
        f, fi = (torch.fft.fft, torch.fft.ifft) if self.dim == 1 else (torch.fft.fft2, torch.fft.ifft2)
        return fi(f(u, norm="ortho") * H, norm="ortho")

    def field(self, u):
        u = self._pad(u)
        for k in range(self.K):
            u = self._fft(self._mask(u, self.P[k]), H=self.H)
        F = (torch.fft.fft if self.dim == 1 else torch.fft.fft2)(self._crop(u), norm="ortho")
        return F

    def forward(self, U):
        F = self.field(U)
        if self.dim == 1:
            return F.abs()[..., self.outputs] ** 2
        return F.abs()[..., self.outputs[0], self.outputs[1]] ** 2

    def scan(self, j):
        """出口の点 j から光を逆向きに 1 回通し（随伴）、入口の面ですべての入口の波との重なりの明るさを返す。"""
        with torch.no_grad():
            N = self.N
            if self.dim == 1:
                e = torch.zeros(N, dtype=torch.complex64); e[self.outputs[j]] = 1
                u = torch.fft.ifft(e, norm="ortho")
            else:
                e = torch.zeros(N, N, dtype=torch.complex64); e[self.outputs[0][j], self.outputs[1][j]] = 1
                u = torch.fft.ifft2(e, norm="ortho")
            u = self._pad(u)
            for k in reversed(range(self.K)):
                u = self._fft(u, H=self.H.conj())                                             # 伝わりを逆向きに
                u = self._mask(u, -self.P[k])                                                 # 板の位相を逆に
            u = self._crop(u)
            return torch.stack([(self.wave(f).conj() * u).sum().abs() ** 2 for f in self.inputs])


def fit(D, U, tg, steps=1500):
    opt = torch.optim.Adam([D.P], lr=0.03); q = torch.cat([tg, torch.zeros(*tg.shape[:-1], 1)], -1)
    for _ in range(steps):
        W = D.forward(U); p = torch.cat([W, (1 - W.sum(-1, keepdim=True)).clamp(min=1e-9)], -1)
        loss = -(q * torch.log(p + 1e-9)).sum(-1).mean(); opt.zero_grad(); loss.backward(); opt.step()


ATTRS = [(v, x) for v in range(NV) for x in range(47)]
g = torch.Generator().manual_seed(7); pick = [ATTRS[int(i)] for i in torch.randperm(len(ATTRS), generator=g)[:60]]
pick = [(v, x) for v, x in pick if TG[:, v, x].max() > 0.05][:20]                             # 走査する属性 20 個（どれかの語が持つもの）
truth_top = {a: set(torch.argsort(TG[:, a[0], a[1]], descending=True)[:5].tolist()) for a in pick}
res = dict(exp="68", words=Wn, attrs=len(pick))
forms = {}
# 1D 共通: 縦 256・板 32 枚
D1 = Device(1, 256, 32, [5 * w + v for w in range(Wn) for v in range(NV)], torch.arange(200, 200 + NX), 64.0, 1)
forms["1D 共通（縦 256・板 32 枚）"] = D1
forms["1D 共通（縦 256・板 64 枚、位相を語ごとの語板とそろえる）"] = Device(1, 256, 64, [5 * w + v for w in range(Wn) for v in range(NV)], torch.arange(200, 200 + NX), 64.0, 3)
# 2D 共通: 32 × 32・板 8 枚
idx = [(w * NV + v) for w in range(Wn) for v in range(NV)]
D2 = Device(2, 32, 8, [((i % 32) - 16, (i // 32) % 32 - 16) for i in idx], (torch.tensor([8 + i // 7 for i in range(NX)]), torch.tensor([8 + i % 7 for i in range(NX)])), 32.0, 2)
forms["2D 共通（32 × 32・板 8 枚）"] = D2
for name, D in forms.items():
    t0 = time.time()
    U = torch.stack([D.wave(f) for f in D.inputs]).view(Wn, NV, *([D.N] if D.dim == 1 else [D.N, D.N]))
    fit(D, U, TG)
    with torch.no_grad():
        Wf = D.forward(U)
    ev = evaluate(pr2(Wf), TE)
    agree, overlap = [], []
    for v, x in pick:
        j = x
        s = D.scan(j).view(Wn, NV)[:, v]                                                      # 逆向き 1 回: 全部の語の、観点 v の入口の明るさ
        fwd = Wf[:, v, j]                                                                     # 順向き（語ごと）
        agree.append(float((s - fwd).abs().max() / fwd.abs().max().clamp(min=1e-12)))
        sc = s / Wf[:, v].sum(-1).clamp(min=1e-12)                                            # 割合に直す（語ごとの光の量で割る）
        overlap.append(len(set(torch.argsort(sc, descending=True)[:5].tolist()) & truth_top[(v, x)]) / 5)
    res[name] = dict(bits=ev["bits"], light=float(Wf.sum(-1).mean()), phases=D.P.numel(), scan_vs_forward_maxrel=max(agree),
                     scan_top5_overlap=sum(overlap) / len(overlap), passes_for_scan=1, sec=time.time() - t0)
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()}, flush=True)
# 語ごとの 1D の語板（実験58 の形）
t0 = time.time(); P = train_1d(TG, torch.ones(TG.shape[:2], dtype=torch.bool), 2000, 0, lr=0.03); Wg = run_1d(P, NV, NX)
ov = [len(set(torch.argsort(pr2(Wg)[:, v, x], descending=True)[:5].tolist()) & truth_top[(v, x)]) / 5 for v, x in pick]
res["語ごとの 1D の語板（縦 64・板 8 枚 × 32 語）"] = dict(bits=evaluate(pr2(Wg), TE)["bits"], light=float(Wg.sum(-1).mean()), phases=P.numel(),
                                                 scan_top5_overlap=sum(ov) / len(ov), passes_for_scan=Wn, sec=time.time() - t0)
print("語ごとの 1D の語板", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["語ごとの 1D の語板（縦 64・板 8 枚 × 32 語）"].items()}, flush=True)
res["表"] = evaluate(TG, TE)["bits"]; res["sec"] = time.time() - t00
with open("results/exp68.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print("表", round(res["表"], 3))
