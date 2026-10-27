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


def test_grid_shadow_lights_matching_cell():
    """実験04: 4 × 4 の周波数の格子（cos・sin の板）に縞を当てると、同じ周波数の区画だけ ±N²/2、ほかは 0。"""
    N = 16
    y, x = np.mgrid[0:N, 0:N]
    u = np.cos(2 * np.pi * (1 * x + 2 * y) / N + np.pi / 2)            # 横 1・縦 2、90° ずらす
    for fy in range(1, 5):
        for fx in range(1, 5):
            B = 2 * np.pi * (fx * x + fy * y) / N
            hit = (fx, fy) == (1, 2)
            assert abs((u * np.cos(B)).sum()) < 1e-9
            assert abs((u * np.sin(B)).sum() - (-N * N / 2 if hit else 0)) < 1e-9


def test_product_shifts_frequency_and_is_circular_conv():
    """実験05: cos(5)·cos(3) は差 2・和 8 に N/4、位相は差・和。FFT(x·g) = (1/N)·（X と G のぐるっと回る畳み込み）。"""
    N = 32
    n = np.arange(N)
    x, g = np.cos(2 * np.pi * 5 * n / N + np.pi / 3), np.cos(2 * np.pi * 3 * n / N + np.pi / 9)
    Y = np.fft.fft(x * g, n=N, norm="backward")
    assert abs(Y[2] - 8 * np.exp(1j * (np.pi / 3 - np.pi / 9))) < 1e-12
    assert abs(Y[8] - 8 * np.exp(1j * (np.pi / 3 + np.pi / 9))) < 1e-12
    X, G = np.fft.fft(x, n=N, norm="backward"), np.fft.fft(g, n=N, norm="backward")
    C = np.array([sum(X[m] * G[(k - m) % N] for m in range(N)) for k in range(N)])
    assert np.abs(Y - C / N).max() < 1e-12


def test_circular_conv_and_zero_padding():
    """実験07: FFT の掛け算はぐるっと回る畳み込み。長さ 2N にゼロを詰めると普通の畳み込みに一致する。"""
    N = 8
    x = np.arange(1, N + 1, dtype=float); h = np.zeros(N); h[:3] = 1
    circ = np.fft.ifft(np.fft.fft(x, n=N) * np.fft.fft(h, n=N), n=N).real
    assert np.allclose(circ, [16, 11, 6, 9, 12, 15, 18, 21])
    pad = np.fft.ifft(np.fft.fft(x, n=2 * N) * np.fft.fft(h, n=2 * N), n=2 * N).real[:N + 2]
    assert np.allclose(pad, np.convolve(x, h[:3]))


def test_correlation_peak_at_shift():
    """実験07: ifft(conj(FFT(k))·FFT(y)) は相関 Σ k[n]·y[n + s]。y = k を 3 ずらしたものなら s = 3 に N の山。"""
    N = 8
    k = np.random.default_rng(0).choice([-1.0, 1.0], N); y = np.roll(k, 3)
    c = np.fft.ifft(np.conj(np.fft.fft(k, n=N)) * np.fft.fft(y, n=N), n=N).real
    assert np.argmax(c) == 3 and abs(c[3] - N) < 1e-12


def test_complex_grad_is_real_plus_i_imag():
    """実験10: torch の複素数の勾配は ∂L/∂x + i·∂L/∂y。|z|² なら 2z。"""
    z0 = torch.randn(8, dtype=torch.complex128, generator=torch.Generator().manual_seed(0))
    z = z0.clone().requires_grad_(True)
    (z.abs() ** 2).sum().backward()
    assert (z.grad - 2 * z0).abs().max() < 1e-12
