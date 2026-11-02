"""waves.py — 記号を「波」にする・波から明るさを読む、いちばん下の道具

Sin-CNN3 では、記号（リンゴ・「赤」など）を **整数の周波数を持つ波** で表します。

    記号 s を周波数 f の波にする:   u[y] = exp(2πi · f · y / N) / √N     （y = 0, 1, …, N−1）

なぜ波にするのか:
    - 整数の周波数の波どうしは **直交** します（内積が 0）。だから何本重ねても、あとで
      フーリエ変換（レンズ）で「どの周波数が・どれだけの明るさで入っていたか」を分けて読めます。
    - 周波数を少し **ずらす** と、元の波とも直交する別の波になります。Sin-CNN3 はこれを
      「観点（何を聞くか: 色・形・味…）」に使います（観点周波数、実験21）。

明るさ（|振幅|²）の約束:
    - ここでの波は長さ 1（明るさの合計 1）にそろえています。
    - レンズ（FFT, norm="ortho"）は光の量を保つので、出口の明るさの合計も 1 のままです。
      だから「ある周波数の明るさ」は、そのまま「光のうち何割がそこへ来たか」= 割合・確率として読めます。

このファイルの関数は、どれも torch のテンソル（複素数）を受け取り・返します。
"""
import math

import torch


def wave(freqs, n):
    """整数の周波数の波を作る。

    引数:
        freqs: 周波数（整数）1 つ、または整数のリスト。負の数も使えます（−f は n − f と同じ）。
        n:     波の長さ（画素の数）。読み分けられる周波数は 0〜n−1 の n 個です。
    戻り値:
        complex64 のテンソル。freqs が 1 つなら形 (n,)、リストなら形 (len(freqs), n)。
        どの波も長さ 1（明るさの合計 1）。

    例:
        >>> u = wave(3, 64)          # 周波数 3 の波
        >>> brightness(u)[3]         # レンズの後、周波数 3 に光が全部集まる → 1.0
    """
    single = isinstance(freqs, int)
    fs = torch.tensor([freqs] if single else list(freqs), dtype=torch.float32)
    y = torch.arange(n, dtype=torch.float32)
    u = torch.exp(1j * 2 * math.pi * fs[:, None] * y[None, :] / n) / math.sqrt(n)
    u = u.to(torch.complex64)
    return u[0] if single else u


def lens(field):
    """レンズ = フーリエ変換（光の量を保つ "ortho" の約束）。最後の軸を周波数に分けます。"""
    return torch.fft.fft(field, dim=-1, norm="ortho")


def brightness(field):
    """レンズを通した後の、周波数ごとの明るさ |振幅|²。

    入力が長さ 1 の波なら、戻り値の合計は 1 です（光の量が保たれるため）。
    """
    return lens(field).abs() ** 2


def overlap(a, b):
    """2 つの光の重なりの明るさ |<a, b>|²（形がそっくりなら 1、直交なら 0）。

    a, b は同じ長さの複素数ベクトル（長さ 1 にそろえてあること）。
    光学では、2 つの光を干渉させたときの「重なり積分」に当たります。
    """
    return float((a.conj() * b).sum().abs() ** 2)
