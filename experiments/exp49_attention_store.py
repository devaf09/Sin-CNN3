"""実験49: 光の注意機構 — 鍵と値の置き方（番地 / 周波数の帯 / ホログラムの重ね）と重み付け、文脈の長さ

着想: 2026-10-28 持ち主の案（問いの光と鍵の光の重なりで前の語を足し合わせる注意機構を光で作る。過去の鍵・値を「別の画素（番地）に並べて
      4f 相関器で引く」か「別の空間周波数（搬送波）に多重化してホログラフィックに引き抜く」か、いくつかの条件で比べたい。
      間の語がどれだけ無関係でも、干渉の直交性で無視できて、文脈が長くても埋もれないかも）。部品は src/attention.py（差し替えできる）。
課題: 連想の呼び出し。文脈 = T 個の (鍵, 値)。問い = その中の 1 つの鍵（そのまま、または位相の雑音入り）。正しい値を答えられるか。
      鍵の作り: でたらめ（位相がでたらめな複素数）/ 直交（位置ごとに別の整数の周波数の波。T ≤ N のときだけ）。
      値 = 語彙 64 個の中のどれか（値の光もでたらめな位相の模様）。答え = 足し合わせた場にいちばん近い値の光。
置き方: address（番地に並べる、記憶 T × N）/ band（1 つの場を T 個の周波数の帯に分ける）/ holo（1 枚の板に重ねる）。
重み付け: softmax β = 2・8・32 / べき p = 2・4 / スパイク。比べ: トランスフォーマーそのもの（実数の内積 / √N の softmax、実数のでたらめな鍵）。
式の予想: 重なりの明るさは 正しい位置 1、でたらめな外れ 平均 1/N。softmax では外れも e^{β/N}≈1 の重みを持つので、正しい位置の重みは
          e^β/(e^β + T − 1)。T > e^β で薄まる（直交した鍵で外れの明るさが 0 でも、softmax では 0 にならない）。べき・スパイクでは 0 の外れは 0。
          holo は混線（分散 T/N）、band は鍵が短くなる（長さ N/T）ので、T が N に近づくと崩れる。
指標: 正解率、正しい位置の重み（平均）。3 seed、問い 256。結果の置き場: results/exp49.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp49_attention_store.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention import Attention, dot_softmax

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
VOC, Q = 64, 256
SEL = [("softmax", dict(beta=2.0)), ("softmax", dict(beta=8.0)), ("softmax", dict(beta=32.0)), ("power", dict(p=2)), ("power", dict(p=4)), ("spike", {})]
out = "results/exp49.txt"
for N in [64, 256]:
    for seed in range(3):
        g = torch.Generator().manual_seed(seed)
        rnd = lambda *s: torch.exp(2j * math.pi * torch.rand(*s, generator=g)) / math.sqrt(s[-1])
        Vbook = rnd(VOC, N)
        for T in [4, 16, 64, 256, 1024, 4096]:
            vid = torch.randint(0, VOC, (T,), generator=g); V = Vbook[vid]; idx = torch.randint(0, T, (Q,), generator=g)
            for keykind in ["でたらめ", "直交"]:
                if keykind == "直交" and T > N:
                    continue
                K = rnd(T, N) if keykind == "でたらめ" else torch.exp(2j * math.pi * torch.randperm(N, generator=g)[:T, None] * torch.arange(N)[None, :] / N) / math.sqrt(N)
                for noise in [0.0, 1.0]:                                         # 問いの位相の雑音（1.0 で重なり e^{−1/2}² ≈ 0.37）
                    q = K[idx] * torch.exp(1j * noise * torch.randn(Q, N, generator=g))
                    row = dict(exp="49", N=N, T=T, seed=seed, key=keykind, noise=noise)
                    for store in ["address", "band", "holo"]:
                        for sel, kw in SEL:
                            o, w = Attention(store, sel, **kw)(q, K, V)
                            name = f"{store} {sel}{''.join(f' {k}={v:g}' for k, v in kw.items())}"
                            ans = (o @ Vbook.conj().T).abs().argmax(1)
                            row[name] = float((ans == vid[idx]).float().mean()); row[name + " 重み"] = float(w[torch.arange(Q), idx].mean())
                    if keykind == "でたらめ" and noise == 0.0:                     # トランスフォーマーそのもの（実数）
                        Kr = torch.randn(T, N, generator=g); Vr = torch.randn(VOC, N, generator=g)
                        o, w = dot_softmax(Kr[idx], Kr, Vr[vid])
                        row["トランスフォーマー（実数）"] = float(((o @ Vr.T).argmax(1) == vid[idx]).float().mean())
                        row["トランスフォーマー（実数） 重み"] = float(w[torch.arange(Q), idx].mean())
                    with open(out, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    print(N, seed, T, keykind, noise, {k: round(v, 2) for k, v in row.items() if isinstance(v, float) and "重み" not in k and k != "noise"}, flush=True)
