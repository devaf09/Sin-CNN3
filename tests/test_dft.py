"""確かめた約束のテスト（M0）。ライブラリの版が変わっても気づけるように。実行: python -m pytest tests"""
import math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naive_dft import naive_dft  # noqa: E402


def test_cos8_matches_formula_and_libraries():
    """実験01: 長さ 8 の cos（1 周）は X[1] = X[7] = 4、ほかは 0。素朴な和・numpy・torch が一致する。"""
    N = 8
    x = np.cos(2 * np.pi * np.arange(N) / N)
    expect = np.zeros(N, complex); expect[1] = expect[7] = N / 2
    Xn = np.array(naive_dft(list(x)))
    Xp = np.fft.fft(x, n=N, norm="backward")
    Xt = torch.fft.fft(torch.tensor(x, dtype=torch.float64), n=N, dim=-1, norm="backward").numpy()
    assert np.abs(Xn - expect).max() < 1e-12
    assert np.abs(Xp - Xn).max() < 1e-12
    assert np.abs(Xt - Xn).max() < 1e-12


def test_norm_scale_roundtrip_parseval():
    """実験02: norm の係数は 1・1/√N・1/N。同じ norm で ifft すれば戻る。エネルギーの比は N・1・1/N。"""
    N = 8
    x = 2 * np.cos(2 * np.pi * 2 * np.arange(N) / N + np.pi / 3)
    Xs = np.array(naive_dft(list(x)))
    for norm, c in (("backward", 1.0), ("ortho", N ** -0.5), ("forward", 1.0 / N)):
        X = np.fft.fft(x, n=N, norm=norm)
        Xt = torch.fft.fft(torch.tensor(x), n=N, dim=-1, norm=norm).numpy()
        assert np.abs(X - c * Xs).max() < 1e-12 and np.abs(Xt - X).max() < 1e-12
        assert np.abs(np.fft.ifft(X, n=N, norm=norm) - x).max() < 1e-12
        assert abs((np.abs(X) ** 2).sum() / (x ** 2).sum() - N * c * c) < 1e-12
    # 振幅 2・位相 π/3 は、X[2] = 2·N/2·e^(iπ/3) としてそのまま読める
    assert abs(Xs[2] - 8 * np.exp(1j * np.pi / 3)) < 1e-12


def test_sin_sign():
    """実験02: sin(2πn/N) は X[1] = −iN/2、X[N−1] = +iN/2。"""
    N = 8
    X = np.fft.fft(np.sin(2 * np.pi * np.arange(N) / N), n=N, norm="backward")
    assert abs(X[1] - (-4j)) < 1e-12 and abs(X[7] - 4j) < 1e-12


def test_leak_matches_geometric_sum():
    """実験03: 半端な周波数 f = 2.5（N = 16）の漏れは、等比数列の和の式と一致する。"""
    N, f = 16, 2.5
    n = np.arange(N)
    g = lambda d: (1 - np.exp(2j * np.pi * d)) / (1 - np.exp(2j * np.pi * d / N))
    X = np.fft.fft(np.cos(2 * np.pi * f * n / N), n=N, norm="backward")
    assert np.abs(X - (0.5 * g(f - n) + 0.5 * g(-f - n))).max() < 1e-12
