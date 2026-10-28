"""実験61: 会話に強い語板 — 会話 × 100 + ほか（Web 小説を含む）の混ぜ方で、名詞 2 万語の語板を作る（GPU）

着想: 実験60（Web 小説を含めて）で、会話の数に 100 倍の重みを掛けて混ぜた表が、会話の確かめ用で会話だけの表と同じ（2.16 対 2.15 ビット）、
      全部の確かめ用でも全部まとめた表に近い（2.10 対 2.07）と分かった。この混ぜ方で語板を作り、表に追いつくか、逆引きが会話らしくなるかを見る。
      Web 小説は持ち主が取得（30 条の 4 の情報解析のみ）。この語板は研究と確かめ専用で、公開しない。
方法: 実験60 と同じ 2 万語・値の置き方（全部の材料で固定）。目標 = 混ぜた数を平らに（k = 20）。実験58 の決まり（出口のずらし、2000 回・学習率 0.03、2 値の初期板）。
      5000 枚ずつ GPU で。seed ごとに別の仕事にして、GPU 2 枚で同時に回す。
確かめ: 会話の確かめ用・全部の確かめ用の交差エントロピー（語板 / 混ぜた表 / 全部重み 1 の表）。
        逆引き: 語板の明るさ P(値 | 名詞) に、名詞の多さ（混ぜた数の合計）を掛けて P(名詞 | 値) にした上位（実験59 の「まれな言い回しに引っ張られる」の直し）。
結果の置き場: results/exp61.txt。語板は非公開フォルダ（Sin-CNN3_private/models/）。
実行: python experiments/exp61_goban_mix.py --device cuda --seed 0（experiments/sweeps/sweep61_s0.sh・sweep61_s1.sh）
"""
import argparse, json, os, runpy, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--chunk", type=int, default=5000)
ap.add_argument("--models", default="D:/AI_git/Sin-CNN3_private/models")
ap.add_argument("--out", default="results/exp61.txt")
a = ap.parse_args(); t00 = time.time()
sys.argv = [sys.argv[0], "--out", os.devnull]                                             # 実験60 の材料の読み込みと表の作り方を借りる（表だけの比べも走るが、結果は捨てる）
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "exp60_domain_mix.py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

tables, pri, SRC, NOUNS, NVIEW, POS = ns["tables"], ns["pri"], ns["SRC"], ns["NOUNS"], ns["NVIEW"], ns["POS"]
sys.stdout.reconfigure(encoding="utf-8"); dev = a.device
W_MIX = {k: (100 if k == "会話" else 1) for k in SRC}
TR, TEc, TEa = tables(W_MIX); TR1, _, _ = tables({k: 1 for k in SRC})
sm = lambda T, k=20: (T + k * pri(T)[None]) / (T.sum(2, keepdim=True) + k)
tg = sm(TR); Ps = []
for i in range(0, len(tg), a.chunk):
    t0 = time.time(); part = tg[i:i + a.chunk]
    Ps.append(train(part, torch.ones(part.shape[:2], dtype=torch.bool), a.steps, a.seed * 1000 + i, lr=0.03, dev=dev).cpu())
    print(f"  {i + len(part)} / {len(tg)} 枚  {time.time() - t0:.0f} 秒", flush=True)
P = torch.cat(Ps); W = torch.cat([run(P[i:i + a.chunk].to(dev), len(NVIEW), tg.shape[2]).cpu() for i in range(0, len(P), a.chunk)]); Pr = prob(W)
os.makedirs(a.models, exist_ok=True); torch.save(dict(P=P, words=NOUNS, pos=POS, mix=W_MIX), f"{a.models}/goban_mix100_seed{a.seed}.pt")
row = dict(exp="61", seed=a.seed, mix=W_MIX, nouns=len(NOUNS), steps=a.steps, device=dev,
           会話={"語板": evaluate(Pr, TEc)["bits"], "混ぜた表": evaluate(tg, TEc)["bits"], "全部重み 1 の表": evaluate(sm(TR1), TEc)["bits"]},
           全部={"語板": evaluate(Pr, TEa)["bits"], "混ぜた表": evaluate(tg, TEa)["bits"], "全部重み 1 の表": evaluate(sm(TR1), TEa)["bits"]}, 光=float(W.sum(-1).mean()))
size = TR.sum((1, 2)); rev = {}; rev_raw = {}
for v, x in [("を", "見る"), ("に", "乗る"), ("を", "食べる"), ("どんな", "美味しい"), ("を", "飼う"), ("が", "降る"), ("で", "買う"), ("を", "弾く"), ("が", "咲く"), ("を", "読む"), ("どんな", "優しい"), ("を", "着る")]:
    if x in POS[v]:
        j = NVIEW.index(v); b = Pr[:, j, POS[v][x]]
        rev[f"{v} {x}"] = [NOUNS[int(i)] for i in torch.argsort(b * size, descending=True)[:8]]           # P(名詞 | 値) ∝ P(値 | 名詞) P(名詞)
        rev_raw[f"{v} {x}"] = [NOUNS[int(i)] for i in torch.argsort(b, descending=True)[:5]]
row["逆引き（名詞の多さを掛ける）"] = rev; row["逆引き（掛けない）"] = rev_raw; row["sec"] = time.time() - t00
print({k: (({kk: round(vv, 3) for kk, vv in v.items()}) if isinstance(v, dict) and k in ("会話", "全部") else v) for k, v in row.items() if k in ("会話", "全部", "光")}, flush=True)
print("逆引き（多さを掛ける）", rev, flush=True); print("逆引き（掛けない）", rev_raw, flush=True)
with open(a.out, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
