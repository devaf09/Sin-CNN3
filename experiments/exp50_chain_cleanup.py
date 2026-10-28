"""実験50: 非線形のループで、関係を何段たどっても濁らないか（実験17・47 の続き）

着想: 実験17 では、出口の光をそのまま次の入力にして関係を 5 段たどれたが、1 段ごとに光が約 7% 減った。実験47 で、
      「明るさを p 乗して組み直す」ループは、濁った鍵をきれいに戻せると分かった。これを関係のたどりに入れれば、何段でも濁らないか。
      持ち主の案 2（記憶ループ: 板 → 回折 → 非線形の利得媒質 → 板へ戻す）の、関係のたどりへの当てはめ。
方法: 長さ N = 64 の鍵（位相がでたらめ）を M = 256 個、ひと回りの鎖 k_0 → k_1 → … → k_{M−1} → k_0 として覚える。
      1 段 = 問い q と鍵の重なりの明るさ I_i → 重み w（下の 4 通り）→ 次の問い = Σ w_i k_{i+1}（番地に並べた記憶、src/attention.py）。
      装置の不完全さとして、1 段ごとに位相の雑音（σ = 0.6 で重なり 約 0.70、σ = 1.0 で 約 0.37）と光の減り（× 0.93、実験17）を入れる。
        線形      : 1 枚の板に Σ k_{i+1} ⊗ k_i を重ねた行列（ホログラム）を通す。出口の明るさは 1 に戻す（M > N なので重ねすぎ）
        p = 1     : 重み = 明るさ（そのまま）。出口の明るさを 1 に戻す（利得）
        p = 2・4  : 重み = 明るさの p 乗。出口の明るさを 1 に戻す
        スパイク  : いちばん明るい鍵だけ
指標: 段ごとに、いちばん明るい鍵が正しいか（正解率）、正しい鍵との重なり |<k, q>|²、光の量 |q|²。3 seed、出発点 256 個、30 段。
結果の置き場: results/exp50.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp50_chain_cleanup.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from attention import Attention

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, M, Q, HOPS, LOSS = 64, 256, 256, 30, 0.93
out = "results/exp50.txt"
for SIG in [0.6, 1.0]:
 for seed in range(3):
     g = torch.Generator().manual_seed(seed)
     K = torch.exp(2j * math.pi * torch.rand(M, N, generator=g)) / math.sqrt(N)
     nxt = torch.roll(K, -1, 0)                                                    # k_i の次は k_{i+1}
     start = torch.randint(0, M, (Q,), generator=g)
     W = nxt.T @ K.conj()                                                           # 線形: Σ k_{i+1} ⊗ conj(k_i)（N × N の 1 枚）
     for kind in ["線形", "p=1", "p=2", "p=4", "スパイク"]:
         att = None if kind == "線形" else (Attention("address", "spike") if kind == "スパイク" else Attention("address", "power", p=int(kind[2:])))
         q = K[start].clone(); acc, ov, light = [], [], []
         for h in range(1, HOPS + 1):
             q = q * torch.exp(1j * SIG * torch.randn(Q, N, generator=g)) * math.sqrt(LOSS)   # 装置の雑音と光の減り
             if att is None:
                 q = q @ W.T; q = q / q.norm(dim=1, keepdim=True)
             else:
                 o, _ = att(q, K, nxt); q = o / o.norm(dim=1, keepdim=True)          # 利得で明るさを 1 に戻す
             true = (start + h) % M
             I = (q @ K.conj().T).abs() ** 2
             acc.append(float((I.argmax(1) == true).float().mean()))
             ov.append(float((I[torch.arange(Q), true] / (q.abs() ** 2).sum(1)).mean()))
             light.append(float((q.abs() ** 2).sum(1).mean()))
         r = dict(exp="50", seed=seed, kind=kind, N=N, M=M, sigma=SIG, loss=LOSS, acc=acc, overlap=ov, light=light)
         with open(out, "a", encoding="utf-8") as f:
             f.write(json.dumps(r, ensure_ascii=False) + "\n")
         print(SIG, seed, kind, "正解率", [round(acc[i], 2) for i in (0, 4, 9, 29)], "重なり", [round(ov[i], 2) for i in (0, 4, 9, 29)], "光", [round(light[i], 3) for i in (0, 4, 9, 29)], flush=True)
