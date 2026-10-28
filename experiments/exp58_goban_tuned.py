"""実験58: 語板を表に追いつかせる — 値の出口を観点ごとにずらす + 平らにした目標 + 長めの学習（実験57 の続き）

着想: 実験57 で、目標を平らにしても、よく出る名詞では語板 2.56 ビットが平らにした表 2.46 に届かなかった。学習用の目標にも 0.14 ビット届いていない（器が表を写しきれていない）。
      原因の候補: どの観点でも「いちばん多い値」が同じ出口（番号 0）に来る。光の量を保つ器では、直交した入口（観点）から同じ出口へ集められる光の和は 1 以下
      （実験56 の上限）なので、観点どうしで出口を取り合う。→ 観点 v の r 番目の値を出口 (r + 9v) mod 47 に置いて、多い値どうしの出口をずらす。
      試し（seed 0、よく出る名詞）: ずらしで 2.562 → 2.535、さらに 2000 回・学習率 0.03 で 2.476（表 2.456）。
方法: 実験56 の材料・語。目標 = 平らにした表（k = 5）。出口をずらす。よく出る名詞・動詞はまっさら 2000 回（学習率 0.03）、
      まれな名詞はまっさら 2000 回 / 近い語板を写して 300 回。3 seed。
次へ進む条件（docs/plan_large_scale.md 段階 1）: まれな名詞で、平らにした表との差 0.05 ビット以内。
結果の置き場: results/exp58.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp58_goban_tuned.py
"""
import json, os, runpy, sys, time

import torch

sys.argv = [sys.argv[0], "--seeds", "0"]
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "exp56_goban_grow.py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

NTR, NTE, VTR, VTE, prior, FREQ, RARE = ns["NTR"], ns["NTE"], ns["VTR"], ns["VTE"], ns["prior"], ns["FREQ"], ns["RARE"]


def shift(T, step=9):
    """観点 v の r 番目の値（r < 47）を出口 (r + step·v) mod 47 へ。観点ごとの「その他」はそのまま。"""
    out = T.clone()
    for v in range(T.shape[1]):
        out[:, v, [(r + step * v) % 47 for r in range(47)]] = T[:, v, :47]
    return out


NTR, NTE, VTR, VTE = shift(NTR), shift(NTE), shift(VTR), shift(VTE)
smooth = lambda T, pr, k=5: (T + k * pr[None]) / (T.sum(2, keepdim=True) + k)
allu = lambda T: torch.ones(T.shape[:2], dtype=torch.bool)
out = "results/exp58.txt"
for seed in range(3):
    t0 = time.time(); row = dict(exp="58", seed=seed)
    tf = smooth(NTR[FREQ], prior(NTR)); Pf = train(tf, allu(tf), 2000, seed, lr=0.03); Prf = prob(run(Pf, 5, NTR.shape[2]))
    row["よく出る名詞"] = evaluate(Prf, NTE[FREQ])["bits"]; row["よく出る名詞 表"] = evaluate(tf, NTE[FREQ])["bits"]
    T = NTR[RARE]; tr = smooth(T, prior(NTR))
    lp = torch.einsum("rvx,fvx->rf", T, torch.log(Prf.clamp(min=1e-9))) / T.sum((1, 2)).clamp(min=1)[:, None]
    row["まれ まっさら 2000 回"] = evaluate(prob(run(train(tr, allu(tr), 2000, seed + 1, lr=0.03), 5, NTR.shape[2])), NTE[RARE])["bits"]
    row["まれ 近い語板を写して 300 回"] = evaluate(prob(run(train(tr, allu(tr), 300, seed + 1, lr=0.03, P0=Pf[lp.argmax(1)]), 5, NTR.shape[2])), NTE[RARE])["bits"]
    row["まれ 表"] = evaluate(tr, NTE[RARE])["bits"]
    tv = smooth(VTR, prior(VTR), 10)
    row["動詞"] = evaluate(prob(run(train(tv, allu(tv), 2000, seed, lr=0.03), 4, VTR.shape[2])), VTE)["bits"]; row["動詞 表"] = evaluate(tv, VTE)["bits"]
    row["sec"] = time.time() - t0
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(seed, {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}, flush=True)
