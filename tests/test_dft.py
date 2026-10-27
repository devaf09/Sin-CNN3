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
