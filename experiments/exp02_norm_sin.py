"""実験02: norm の 3 通り・sin の符号・振幅と位相・往復・エネルギー — 長さ 8 の波で式と突き合わせる

着想: 学び 01 の残った疑問（2026-10-27 持ち主が OK）。cos の山の高さ 4（= N/2）は norm の約束で変わるはず。sin は虚部に符号つきで出るはず。
方法: N = 8 の 3 つの信号 — cos(2π·n/8)、sin(2π·n/8)、2·cos(2π·2n/8 + π/3)（振幅 2・位相 π/3 を載せたもの）— を
      numpy・torch の fft に norm = "backward" / "ortho" / "forward" で通し、素朴な和に係数（1、1/√N、1/N）を掛けた式の予想と比べる。
      ifft で元に戻るか（同じ norm どうし、違う norm どうし）、エネルギー（2 乗の和）が保たれるか（パーセバルの等式）も見る。
指標: 式の予想との差の最大値、往復の差、エネルギーの比 Σ|X|² / Σ|x|²。
結果の置き場: results/exp02.txt。図は tools/fig_exp02.py → docs/images/exp02_norm_sin.png
実行: CUDA_VISIBLE_DEVICES= python experiments/exp02_norm_sin.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from naive_dft import naive_dft  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)        # 乱数は使わない
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp02.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
t0, N = time.time(), a.N
n = np.arange(N)
signals = {"cos": np.cos(2 * math.pi * n / N), "sin": np.sin(2 * math.pi * n / N),
           "2cos(k=2, φ=π/3)": 2 * np.cos(2 * math.pi * 2 * n / N + math.pi / 3)}
scale = {"backward": 1.0, "ortho": 1 / math.sqrt(N), "forward": 1 / N}   # 変換の側で掛かる係数（式の予想）
res = dict(exp="02", args=vars(a), rows=[])
for name, x in signals.items():
    Xs = np.array(naive_dft(list(x)))                                   # 素朴な和（係数なし）
    print(f"\n== {name}: x = {np.round(x, 3).tolist()}")
    for norm, c in scale.items():
        Xp = np.fft.fft(x, n=N, norm=norm)                              # numpy: 長さ N、norm を明示
        Xt = torch.fft.fft(torch.tensor(x, device=a.device), n=N, dim=-1, norm=norm).cpu().numpy()
        back = np.fft.ifft(Xp, n=N, norm=norm).real                     # 同じ norm で逆変換
        energy = float((np.abs(Xp) ** 2).sum() / (x ** 2).sum())        # パーセバル: backward は N、ortho は 1、forward は 1/N のはず
        big = [(k, round(float(abs(z)), 4), round(math.degrees(math.atan2(z.imag, z.real)), 1)) for k, z in enumerate(Xp) if abs(z) > 1e-9]
        r = dict(signal=name, norm=norm, err_numpy=float(np.abs(Xp - c * Xs).max()), err_torch=float(np.abs(Xt - c * Xs).max()),
                 roundtrip=float(np.abs(back - x).max()), energy_ratio=energy, peaks=big)
        res["rows"].append(r)
        print(f"  {norm:>8}: 山 (k, 大きさ, 位相[度]) = {big}  予想との差 {r['err_numpy']:.1e}/{r['err_torch']:.1e}  "
              f"往復の差 {r['roundtrip']:.1e}  エネルギーの比 {energy:.4f}")
# 違う norm どうしで往復すると、元に戻らない（大きさがずれる）
x = signals["cos"]
mix = {f"fft {f} → ifft {i}": float(np.fft.ifft(np.fft.fft(x, n=N, norm=f), n=N, norm=i).real[0] / x[0])
       for f in scale for i in scale}
print("\n違う norm で往復したときの倍率（元 = 1）:", {k: round(v, 4) for k, v in mix.items()})
res.update(mix_roundtrip=mix, sec=time.time() - t0, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
