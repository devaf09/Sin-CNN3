"""attention.py — 光の注意機構（差し替えできる部品）

持ち主の方針（2026-10-28）: 注意機構は、ひとまずトランスフォーマーの形を借りる（暫定）。いずれもっと良い注意機構に入れ替える。
そのため、注意機構を 3 つの役に分け、それぞれを名前で選んで入れ替えられるようにする。

    1. 置き方（Store）   : 過去の鍵の光をどう持つか、問いの光との重なり（相関）をどう取るか
    2. 重み付け（Select）: 重なりの明るさから、どの位置をどれだけ使うかを決める（softmax・べき・スパイク）
    3. 足し合わせ        : 重みを励起光として、過去の値の光を通し、レンズで集めて 1 つの場に足す

光学系での対応（持ち主の案）:
    問いの光 q   = いま着目している語・文脈を表す入力光の模様（長さ N の複素数の場）
    鍵の光 k_t   = 過去の位置 t の語を表す参照光の模様
    内積         = 2 つの光の干渉・重なり積分（相関）。形が一致するほど明るい。ここでは明るさ |<k_t, q>|² を使う
    重み付け     = 相関の明るい所をスパイク化・softmax など
    値の加重平均 = Σ_t w_t v_t（スパイク光で値の模様を透過させ、レンズで集める）

使い方:
    att = Attention(store="address", select="softmax", beta=8.0)
    out, w = att(q, K, V)        # q: (B, N)、K: (T, N)、V: (T, N)  → out: (B, N)、w: (B, T)
新しい置き方・重み付けは STORES / SELECTS に関数を足すだけで使える。
"""
import math

import torch

# ---------- 1. 置き方: (q, K) → 重なりの明るさ I（B × T、正しい鍵なら約 1、でたらめな鍵なら平均 1/N） ----------


def store_address(q, K):
    """番地に並べる: 鍵ごとに別の画素（番地）に置き、4f 相関器で 1 つずつ引く。鍵どうしは混ざらない（記憶の大きさ T × N）。"""
    return (q @ K.conj().T).abs() ** 2


def store_band(q, K):
    """周波数の帯に分ける: 1 つの場（N 個の周波数）を T 個の帯に分け、位置 t の鍵を帯 t に置く（帯の幅 B = N // T）。
    問いも帯ごとに写して（回折格子で T 個に複製するつもり）、帯 t の中だけで重なりを取る。帯の幅が狭いほど鍵が短くなり、混ざりやすい。
    T > N では置けないので、帯の幅 1 にして回り込ませる（同じ帯に複数の鍵 = 重なる）。"""
    T, N = K.shape
    B = max(N // T, 1)
    Kb = K[:, :B] / K[:, :B].norm(dim=1, keepdim=True)                          # 帯の幅に切った鍵（長さ 1 にそろえる）
    qb = q[:, :B] / q[:, :B].norm(dim=1, keepdim=True)
    if T * B <= N:
        return (qb @ Kb.conj().T).abs() ** 2
    band = torch.arange(T) % N                                                  # 帯が足りない: 同じ帯の鍵は場で足し合わされる
    Ksum = torch.zeros(N, B, dtype=K.dtype).index_add_(0, band, Kb)
    return (qb @ Ksum.conj().T).abs()[:, band] ** 2


def store_holo(q, K):
    """ホログラムに重ねる: 1 枚の板に Σ_t conj(FFT k_t)·(相関面の点 t へのずらし) を重ね書きし、問いを通して相関面の点 t の明るさを読む。
    点 t には正しい重なりのほかに、ほかの鍵からの混線（分散 約 T/N）が乗る（実験08・47）。T > N では点が足りないので点 t mod N に重なる。"""
    T, N = K.shape
    pos = torch.arange(T) % N
    H = (torch.fft.fft(K, dim=1).conj() * torch.exp(-2j * math.pi * pos[:, None] * torch.arange(N)[None, :] / N)).sum(0)
    plane = torch.fft.ifft(torch.fft.fft(q, dim=1) * H, dim=1)                   # 相関面の場
    return plane.abs()[:, pos] ** 2


STORES = {"address": store_address, "band": store_band, "holo": store_holo}

# ---------- 2. 重み付け: I → w（B × T、和 1） ----------


def select_softmax(I, beta=8.0, **_):
    """softmax: w = exp(β·I) / Σ。トランスフォーマーと同じ形（I は明るさなので、正しい鍵 1、外れ 平均 1/N）。
    注意: 重なりが 0 の位置（直交した鍵）も exp(0) = 1 の重みを持つ。正しい位置の重みは e^β / (e^β + T − 1) なので、
    文脈 T が長いと β を ln T より大きくしないと薄まる。"""
    return torch.softmax(beta * I, dim=-1)


def select_power(I, p=2, **_):
    """べき: w = I^p / Σ I^p（実験47 の「明るさを p 乗して出し直す」）。"""
    x = (I / I.max(dim=-1, keepdim=True).values) ** p
    return x / x.sum(dim=-1, keepdim=True)


def select_spike(I, **_):
    """スパイク: いちばん明るい位置だけ 1（p → ∞）。"""
    return torch.nn.functional.one_hot(I.argmax(dim=-1), I.shape[-1]).to(I.dtype)


SELECTS = {"softmax": select_softmax, "power": select_power, "spike": select_spike}


class Attention:
    """光の注意機構。store・select を名前で選ぶ。self.select_kw は重み付けの強さ（beta・p）。"""

    def __init__(self, store="address", select="softmax", **select_kw):
        self.store, self.select, self.select_kw = STORES[store], SELECTS[select], select_kw

    def __call__(self, q, K, V):
        I = self.store(q, K)                                                     # 1. 重なりの明るさ
        w = self.select(I, **self.select_kw)                                     # 2. 重み
        out = w.to(V.dtype) @ V                                                  # 3. 値を重みで足し合わせる（レンズで集める）
        return out, w


def dot_softmax(q, K, V, scale=None):
    """比べ: トランスフォーマーそのもの（実数の内積 / √d の softmax）。q, K, V は実数。"""
    d = K.shape[1]
    w = torch.softmax((q @ K.T) / (scale or math.sqrt(d)), dim=-1)
    return w @ V, w
