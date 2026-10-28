"""実験57: 語板に平らにした表を学ばせる（大規模な計画の段階 1）

着想: 実験56 で、まれな名詞では「表を観点ごとの割合へ寄せて平らにする」（数の少ない語ほど全体の割合に寄せる、昔からの平滑化）が 2.55 ビットで、
      語板（近い語板を写して 100 回）の 2.70 より良かった。語板は学ぶ目標の表をまねるので、目標そのものを平らにした表にすれば追いつくはず。
方法: 実験56 と同じ材料・語・器（「その他」は観点ごと）。語板の目標 = (数 + k × 観点ごとの割合) / (数の和 + k)、k = 0（実験56）・2・5・10。
      まれな名詞（401〜1200 番目）はまっさら 600 回と近い語板を写して 100 回、よく出る名詞（上位 400）と動詞（300）はまっさら 600 回。3 seed。
次へ進む条件（docs/plan_large_scale.md）: まれな名詞で、平らにした表との差 0.05 ビット以内。
結果の置き場: results/exp57.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp57_goban_smooth.py
"""
import json, os, runpy, sys, time

import torch

sys.argv = [sys.argv[0], "--seeds", "0"]                                                  # 実験56 の材料・語・関数を借りる（学習はしない）
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "exp56_goban_grow.py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

NTR, NTE, VTR, VTE, prior, FREQ, RARE, NV = ns["NTR"], ns["NTE"], ns["VTR"], ns["VTE"], ns["prior"], ns["FREQ"], ns["RARE"], ns["NV"]
nn, nvv = NTR.shape[2], VTR.shape[2]
smooth = lambda T, pr, k: (T + k * pr[None]) / (T.sum(2, keepdim=True) + k)
out = "results/exp57.txt"
for seed in range(3):
    for k in [0, 2, 5, 10]:
        t0 = time.time(); row = dict(exp="57", seed=seed, k=k)
        tf = smooth(NTR[FREQ], prior(NTR), k) if k else NTR[FREQ] / NTR[FREQ].sum(2, keepdim=True).clamp(min=1)
        Pf = train(tf, NTR[FREQ].sum(2) >= (1 if k else 3), 600, seed); Prf = prob(run(Pf, 5, nn))
        row["よく出る名詞"] = evaluate(Prf, NTE[FREQ])
        T = NTR[RARE]; tr = smooth(T, prior(NTR), k) if k else T / T.sum(2, keepdim=True).clamp(min=1)
        lp = torch.einsum("rvx,fvx->rf", T, torch.log(Prf.clamp(min=1e-9))) / T.sum((1, 2)).clamp(min=1)[:, None]
        row["まれ まっさら 600 回"] = evaluate(prob(run(train(tr, T.sum(2) >= 1 if not k else torch.ones(T.shape[:2], dtype=torch.bool), 600, seed + 1), 5, nn)), NTE[RARE])
        row["まれ 近い語板を写して 100 回"] = evaluate(prob(run(train(tr, T.sum(2) >= 1 if not k else torch.ones(T.shape[:2], dtype=torch.bool), 100, seed + 1, P0=Pf[lp.argmax(1)]), 5, nn)), NTE[RARE])
        tv = smooth(VTR, prior(VTR), k) if k else VTR / VTR.sum(2, keepdim=True).clamp(min=1)
        row["動詞"] = evaluate(prob(run(train(tv, VTR.sum(2) >= (1 if k else 3), 600, seed), 4, nvv)), VTE)
        row["表（同じ k で平らに）"] = {"よく出る名詞": evaluate(smooth(NTR[FREQ], prior(NTR), max(k, 0.5)), NTE[FREQ])["bits"],
                                  "まれ": evaluate(smooth(T, prior(NTR), max(k, 0.5)), NTE[RARE])["bits"], "動詞": evaluate(smooth(VTR, prior(VTR), max(k, 0.5)), VTE)["bits"]}
        row["sec"] = time.time() - t0
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(seed, "k", k, {kk: round(v["bits"], 3) for kk, v in row.items() if isinstance(v, dict) and "bits" in v}, "表", {kk: round(v, 3) for kk, v in row["表（同じ k で平らに）"].items()}, f"{row['sec']:.0f} 秒", flush=True)
