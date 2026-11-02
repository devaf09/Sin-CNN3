"""device.py — 光の器（位相の板 + 回折を何枚も重ねたもの）

光の器のしくみ（図: docs/images/overview_device.png）:

    入口の光 ──▶ [板 1] ─回折─▶ [板 2] ─回折─▶ … ─▶ [板 K] ─回折─▶ レンズ ──▶ 出口の明るさ

    - 板（plate）: 画素ごとに光の位相だけを変える薄い板。位相 θ の画素を通ると、光は exp(iθ) 倍になる。
                   この位相が **学習する数** です（画素 n × 板 K 枚）。
    - 回折（diffraction）: 板と板のあいだを光が進むと、模様が広がって混ざり合う。
                   「角スペクトル法」で計算します（周波数ごとに位相を回すだけ。光の量は変わらない）。
    - レンズ: 最後にフーリエ変換して、周波数ごとの明るさを読む（waves.brightness）。

大事な性質:
    1. **光の量が保たれる（ユニタリ）**。板も回折も「位相を変えるだけ」なので、光は増えも減りもしない。
       だから出口の明るさの合計 ≤ 1 で、明るさをそのまま「割合」として読めます。
       その裏返しで「違う入口から同じ出口へ集められる光の和は 1 まで」という決まりもあります（実験56〜58）。
    2. **逆向きに通せる**（backward）。各段の逆を逆の順に行うだけ。
       入口 g → 出口 f の明るさと、出口 f → 入口 g の明るさは **ぴったり等しい**（実験21）。
       これが「逆引き」（値 → それを持つ物）の土台です。
    3. 板の端から出た光が反対側から戻ってこないよう、計算の場を 2 倍に広げて 0 を詰めています
       （前のプロジェクトの「ぐるっと回る計算」の食い違いを避けるため。docs/lessons/00 を参照）。

この器は回折の深層ネットワーク（D²NN, 2018）と同じ形です。新しいのは器ではなく、使い方（knowledge.py）。
"""
import math

import torch

from .waves import lens


class LightDevice:
    """光の器 1 台。

    引数:
        n:      縦の画素の数（= 読み分けられる周波数の数）。既定 64。
        plates: 板の枚数 K。器を大きくするなら、縦より板の枚数を増やすのが割に合う（実験75）。既定 8。
        gap:    板と板のあいだの距離（画素の単位、波長 1）。既定 64。
        seed:   初期の板を作る乱数の種。
        init:   初期の板の形。"binary"（各画素 0 か π、学習がいちばん速い。実験45・46）か "flat"（全部 0）。

    主な使い方:
        dev = LightDevice(n=64, plates=8)
        out = dev.forward(u_in)          # 入口の光 (B, n) → 出口の光 (B, 2n)（レンズの前）
        I   = dev.brightness(u_in)       # 出口の明るさ (B, n)
        u   = dev.backward(out)          # 逆向き: 出口の光 → 入口の光 (B, n)
    """

    def __init__(self, n=64, plates=8, gap=64.0, seed=0, init="binary", phase=None):
        self.n, self.plates, self.gap = n, plates, gap
        self.L = 2 * n                                                    # 0 を詰めて 2 倍に広げた計算の場
        fx = torch.fft.fftfreq(self.L)                                    # 場の周波数（波長 1 の単位）
        # 角スペクトル法の伝達関数: 距離 gap 進むと、周波数 fx の成分は位相 gap·√((2π)² − (2π fx)²) だけ回る
        self.H = torch.exp(1j * gap * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * fx) ** 2)).to(torch.complex64)
        if phase is not None:
            p0 = phase.detach().clone()
        else:
            g = torch.Generator().manual_seed(seed)
            if init == "binary":
                p0 = math.pi * (torch.rand(n, plates, generator=g) < 0.5).float()
            elif init == "flat":
                p0 = torch.zeros(n, plates)
            else:
                raise ValueError(f"init は 'binary' か 'flat': {init}")
        self.phase = torch.nn.Parameter(p0)                               # 学習する数: 画素ごとの位相 (n, plates)

    # ---- 1 段ずつの部品 ----
    def _plate(self, u, k, sign=1):
        """板 k を通す（sign=−1 なら逆の位相 = 板の逆）。0 を詰めた側（u[:, n:]）は板の外なので素通り。"""
        n = self.n
        return torch.cat([u[:, :n] * torch.exp(sign * 1j * self.phase[:, k]), u[:, n:]], 1)

    def _diffract(self, u, sign=1):
        """距離 gap だけ進む（sign=−1 なら逆向きに戻る）。"""
        H = self.H if sign == 1 else self.H.conj()
        return torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")

    # ---- 順向き・逆向き ----
    def forward(self, u_in):
        """入口の光 (B, n) または (n,) を通して、出口の光 (B, 2n)（レンズの前）を返す。"""
        u = u_in[None] if u_in.dim() == 1 else u_in
        u = torch.nn.functional.pad(u.to(torch.complex64), (0, self.L - self.n))
        for k in range(self.plates):
            u = self._diffract(self._plate(u, k))
        return u

    def backward(self, u_out):
        """出口の光 (B, 2n) または (B, n) を逆向きに通し、入口の光 (B, n) を返す（各段の逆を逆の順に）。"""
        u = u_out[None] if u_out.dim() == 1 else u_out
        if u.shape[-1] == self.n:
            u = torch.nn.functional.pad(u.to(torch.complex64), (0, self.L - self.n))
        for k in reversed(range(self.plates)):
            u = self._plate(self._diffract(u, -1), k, -1)
        return u[:, :self.n]

    def brightness(self, u_in):
        """入口の光を通し、レンズの後の明るさ (B, n) を返す（出口は場の前半 n 画素を読む）。"""
        return lens(self.forward(u_in)[:, :self.n]).abs() ** 2

    def copy(self):
        """板を写した新しい器（学習の出発点に使う。似た物の器を写すと覚えが速い、実験51・56）。"""
        return LightDevice(self.n, self.plates, self.gap, phase=self.phase)
