"""naive_dft.py — 素朴な離散フーリエ変換（自作の基準）

式をそのまま書いたもの。速さは遅い（N² 回の掛け算）が、式と 1 行ずつ対応するので、中身を目で確かめられる。
    X[k] = Σ_{n=0}^{N-1} x[n]·e^(−2πi·kn/N)        （k = 0, 1, …, N−1）
「k 番目の周波数の時計の針（e^(−2πi·kn/N)）を、信号の各点に掛けて全部足す」。信号の中に、その速さで回る成分があれば、
足し合わせが打ち消し合わずに大きく残る。正規化（1/N などで割ること）はしない（numpy・torch の norm="backward" と同じ約束）。
"""
import cmath


def naive_dft(x):
    """x: 数の並び（実数でも複素数でも）。戻り値: 長さ N の複素数のリスト X[0..N-1]。"""
    N = len(x)
    X = []
    for k in range(N):
        s = 0j
        for n in range(N):
            s += x[n] * cmath.exp(-2j * cmath.pi * k * n / N)   # 時計の針を掛けて足す
        X.append(s)
    return X
