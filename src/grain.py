"""grain.py — 1 つの語（物）の学習器「グレイン」（仮の名前。持ち主の呼び方。名前の候補は docs/lessons/51 を参照）

1 つのグレイン = 1 枚ぶんの光の器（画素ごとの位相の板 K 列 + 列の間の回折、学び 31 の「単語ごとの格子」の画素版）。
入口に「観点の波」（観点 v = 整数の周波数 v の波）を入れると、出口のレンズの後で、その観点の値（属性）の明るさが出る。
    明るさ = その物がその値を持つ割合（例: リンゴ・色 → 赤 0.7・緑 0.2・黄 0.1）
逆引き（値 → 物）の明るさは、順引きの明るさと等しい（器が光の量を保つため、学び 21）。ここでは順引きの明るさを使う。

使い方:
    g = Grain(n_views=5, n_vals=14, seed=0)
    g.fit(target, steps=300)           # target: (観点の数, 値の数) の割合の表（行ごとに和 1。その観点で使わない値は 0）
    P = g.answer()                     # (観点の数, 値の数) の明るさ
    g2 = g.copy()                      # 板をそのまま写して、新しいグレインの出発点にする
"""
import copy, math

import torch

N, K, DZ = 64, 8, 64.0                                                          # 縦の画素、板の列、列の間
VAL0 = 10                                                                        # 値 j は周波数 VAL0 + j に読む（観点は 1〜9）
_L = 2 * N
_H = torch.exp(1j * DZ * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(_L)) ** 2))
_y = torch.arange(N, dtype=torch.float32)


def view_wave(views):
    """観点の波（観点 v = 周波数 v）。views: 観点の番号のリスト（1 から）。"""
    return torch.stack([torch.exp(1j * 2 * math.pi * v * _y / N) / math.sqrt(N) for v in views]).to(torch.complex64)


class Grain:
    def __init__(self, n_views, n_vals, seed=0, plate=None, waves=None, val0=VAL0):
        """waves: 入口の周波数のリスト（省略すると観点 1〜n_views）。val0: 値 0 を読む周波数。"""
        self.nv, self.nx, self.val0 = n_views, n_vals, val0
        g = torch.Generator().manual_seed(seed)
        P0 = math.pi * (torch.rand(N, K, generator=g) < 0.5).float() if plate is None else plate.clone()   # 2 値（0 か π）の初期板（学び 45 で最も速い）
        self.P = torch.nn.Parameter(P0)
        self.waves = list(range(1, n_views + 1)) if waves is None else list(waves)
        self.u0 = view_wave(self.waves)

    def _run(self, P):
        u = torch.nn.functional.pad(self.u0, (0, _L - N))
        for k in range(K):
            u = torch.cat([u[:, :N] * torch.exp(1j * P[:, k]), u[:, N:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * _H, dim=-1, norm="ortho")
        return torch.fft.fft(u[:, :N], dim=-1, norm="ortho").abs()[:, self.val0:self.val0 + self.nx] ** 2

    def answer(self):
        """各観点で、各値に届いた光の明るさ（観点の数 × 値の数）。"""
        with torch.no_grad():
            return self._run(self.P)

    def fit(self, target, steps=300, lr=0.01, mask=None):
        """target の割合に合わせて板を学習する。mask: 学習に使う観点（True の行だけ）。戻り値: 25 回ごとの表との差。"""
        opt = torch.optim.Adam([self.P], lr=lr); curve = []
        m = torch.ones(self.nv, dtype=torch.bool) if mask is None else mask
        for s in range(steps + 1):
            W = self._run(self.P)
            if s % 25 == 0:
                curve.append(tv(W.detach(), target, m))
            if s == steps:
                break
            p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
            q = torch.cat([target, torch.zeros(self.nv, 1)], 1)
            loss = -(q[m] * torch.log(p[m] + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        return curve

    def copy(self):
        return Grain(self.nv, self.nx, plate=self.P.detach(), waves=self.waves, val0=self.val0)


def tv(W, target, mask=None):
    """表との差（全変動距離。観点ごとに明るさを割合に直して比べ、平均）。"""
    m = torch.ones(len(target), dtype=torch.bool) if mask is None else mask
    Wn = W / W.sum(1, keepdim=True).clamp(min=1e-12)
    return float((Wn - target).abs().sum(1)[m].mean() / 2)
