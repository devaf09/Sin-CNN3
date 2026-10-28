"""goban_batch.py — 語板（1 語 = 1 枚の光の器）をまとめて学習・評価する部品（実験55 の中身を部品にしたもの）

器は src/grain.py と同じ（縦 N = 64、画素の板 K = 8 列、列の間 64 画素）。観点 v（1 から）= 周波数 v の波を入れ、
値 j の明るさを周波数 VAL0 + j で読む。たくさんの語板を 1 つのテンソル P（語, N, K）にまとめて計算する。

使い方:
    P = train(target, use, steps=600, seed=0)          # target: (語, 観点, 値) の割合、use: (語, 観点) 学習に使うか
    W = run(P, n_views, n_vals)                        # (語, 観点, 値) の明るさ
    ev = evaluate(prob, counts)                        # 確かめ用の数 counts で、交差エントロピー・上位 1・5
"""
import math

import torch

from grain import N, K, DZ, view_wave

VAL0 = 10
_L = 2 * N
_H = {}


def _h(dev):
    if dev not in _H:
        _H[dev] = torch.exp(1j * DZ * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(_L)) ** 2)).to(dev)
    return _H[dev]


def run(P, n_views, n_vals):
    dev = P.device; H = _h(dev); u0 = view_wave(range(1, n_views + 1)).to(dev)
    u = torch.nn.functional.pad(u0, (0, _L - N))[None].expand(P.shape[0], -1, -1)
    for k in range(K):
        u = torch.cat([u[..., :N] * torch.exp(1j * P[:, None, :, k]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
    return torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs()[..., VAL0:VAL0 + n_vals] ** 2


def init_plates(n, seed=0, dev="cpu"):
    """2 値（0 か π）の初期板（学び 45 で最も速い）。"""
    g = torch.Generator().manual_seed(seed)
    return (math.pi * (torch.rand(n, N, K, generator=g) < 0.5).float()).to(dev)


def train(target, use, steps=600, seed=0, P0=None, lr=0.01, dev="cpu"):
    """target の割合に合わせて語板をまとめて学習する。P0 を渡すとそこから始める（近い語板を写したときなど）。"""
    nv, nx = target.shape[1], target.shape[2]
    P = torch.nn.Parameter((init_plates(target.shape[0], seed, dev) if P0 is None else P0.clone()).to(dev))
    opt = torch.optim.Adam([P], lr=lr); q = torch.cat([target, torch.zeros(*target.shape[:2], 1)], -1).to(dev); m = use.to(dev)
    for _ in range(steps):
        W = run(P, nv, nx); p = torch.cat([W, (1 - W.sum(-1, keepdim=True)).clamp(min=1e-9)], -1)
        loss = -(q * torch.log(p + 1e-9)).sum(-1)[m].mean(); opt.zero_grad(); loss.backward(); opt.step()
    return P.detach()


def prob(W, eps=0.02):
    """明るさを割合に直し、少しだけ平らにする（0 の値を避ける）。"""
    return (1 - eps) * W / W.sum(-1, keepdim=True).clamp(min=1e-12) + eps / W.shape[-1]


def evaluate(Pr, counts):
    """counts: (語, 観点, 値) の確かめ用の数。数で重みをつけた交差エントロピー（ビット）と上位 1・5。"""
    Pr = Pr.cpu(); w = counts; lp = torch.log2(Pr.clamp(min=1e-9))
    rank = (Pr[..., None, :] > Pr[..., :, None]).sum(-1)
    s = w.sum().clamp(min=1)
    return dict(bits=float(-(w * lp).sum() / s), top1=float((w * (rank < 1)).sum() / s), top5=float((w * (rank < 5)).sum() / s), n=int(w.sum()))
