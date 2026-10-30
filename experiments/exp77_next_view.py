"""実験77: 「隣の文に出る語」の観点 — 語板に、すぐ前・すぐ後の文に一緒に出る名詞を覚えさせる（談話の観点）

着想: 実験76d〜h で、属性変位（B と D から間の文 C の名詞を当てる）の目安（数え方の表 = 隣の文に一緒に出た名詞）が、
      会話 0.48・小説 0.25 と、語板の方法（0.15・0.07）よりずっと上だった。語板は「同じ文の中の使われ方」しか持っていないので、
      「隣の文に一緒に出る」つながりを観点として足す（持ち主 2026-10-30「始めて」）。
困ったこと: 隣の文に出る名詞は何万種類もあり、出口（値 100 + その他）に入りきらない。全体の上位 100 語にすると「人・事・時」ばかりになる。
方法: 名詞を、使われ方の似た仲間 K = 100 組に分ける（語板 v3 の属性の明るさで k 平均法）。観点「隣」の値 = 隣の文に出た名詞の**仲間の番号**。
      語板（縦 128 × 板 16、観点 1 つ = 周波数 1、値 100 + その他 1）を、仲間の割合（平らにした表、k = 20）に学ばせる。500 回。
      候補の語 y の点 = Σ_{x ∈ B・D の名詞} log P(y の仲間 | x の語板)（仲間の中では語の多さは「多さをそろえた比べ」で打ち消される）。
      比べ: 同じ仲間の割合を、光を使わずに数えた表（仲間の数え方）。語ごとの数え方（実験76 の目安）。
      当たりの測り方は実験76 と同じ（多さをそろえた MRR）。評価は experiments/exp76_attribute_shift.py に NEXT=1 で足す。
材料: 実験76 の目安の表（非公開フォルダの data/views/adjacent_<SOURCE>.pkl。会話 = 学習用の側、小説 = 問題に使わない青空文庫の全作品）。
出力: 非公開フォルダの models/goban_next_<SOURCE>.pt（板・語・仲間の番号・数え方の割合）、results/exp77.txt
実行: python experiments/exp77_next_view.py --device cuda --source conv（experiments/sweeps/sweep77_next_conv.sh・aozora.sh）
"""
import argparse, json, os, pickle, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="conv")
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--k", type=int, default=100)                                             # 仲間の数（= 観点「隣」の値の数）
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run, train, prob

sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time(); dev = a.device; PRIV = "D:/AI_git/Sin-CNN3_private"
# ---- 語板 v3 の語と、属性の明るさ（仲間分けに使う） ----
Ps = []; WORDS = []
for k in (0, 1):
    M = torch.load(f"{PRIV}/models/goban_v3_part{k}_scratch.pt", weights_only=False); Ps.append(M["P"]); WORDS += M["words"]
WI = {w: i for i, w in enumerate(WORDS)}
with torch.no_grad():
    P = torch.cat(Ps); B3 = torch.cat([run(P[i:i + 1500].to(dev), 9, 54).cpu() for i in range(0, len(WORDS), 1500)])
FL = (B3 / B3.sum(-1, keepdim=True).clamp(min=1e-12))[:, :, :45].reshape(len(WORDS), -1)   # 1 語 = 405 個の属性の明るさ
X = torch.nn.functional.normalize(FL - FL.mean(0), dim=1)                                  # 平均を引いて長さをそろえる（cos で近さを測る）
# ---- k 平均法で K 組の仲間に分ける ----
g = torch.Generator().manual_seed(0)
C = X[torch.randperm(len(WORDS), generator=g)[:a.k]].clone()                                # 最初の中心 = でたらめに選んだ語
for it in range(30):
    lab = (X @ C.T).argmax(1)                                                               # いちばん近い中心の組
    for c in range(a.k):
        m = lab == c
        if m.any():
            C[c] = torch.nn.functional.normalize(X[m].mean(0), dim=0)
sizes = torch.bincount(lab, minlength=a.k)
print("仲間分け", a.k, "組（大きさ 最小", int(sizes.min()), "最大", int(sizes.max()), "）", f"{time.time() - t0:.0f} 秒", flush=True)
# ---- 隣の文に出た名詞の仲間の数（語 × 仲間） ----
ADJ = pickle.load(open(f"{PRIV}/data/views/adjacent_{a.source}.pkl", "rb"))["N"]
T = torch.zeros(len(WORDS), a.k + 1)                                                        # 最後の 1 つは「その他」（使わないが、器の決まりで置く）
for x, row in ADJ.items():
    i = WI.get(x)
    if i is None:
        continue
    for y, m in row.items():
        j = WI.get(y)
        if j is not None:
            T[i, int(lab[j])] += m
has = T.sum(1) > 0
pri = (T.sum(0) + 0.5) / (T.sum(0) + 0.5).sum()                                             # 全体の仲間の割合
tg = (T + 20 * pri) / (T.sum(1, keepdim=True) + 20)                                         # 平らにした表（数の少ない語は全体へ寄せる）
print("隣の数", int(T.sum()), "数のある語", int(has.sum()), f"{time.time() - t0:.0f} 秒", flush=True)
# ---- 語板（縦 128 × 板 16、観点 1 つ）に学ばせる ----
B = 1500; Pn = []
for i in range(0, len(WORDS), B):
    Pn.append(train(tg[i:i + B, None, :], torch.ones(tg[i:i + B].shape[0], 1, dtype=torch.bool), a.steps, i, lr=0.03, dev=dev, rows=128, plates=16).cpu())
Pn = torch.cat(Pn)
W = torch.cat([run(Pn[i:i + B].to(dev), 1, a.k + 1).cpu() for i in range(0, len(WORDS), B)])[:, 0]
pb = prob(W)                                                                                 # 語板の答え（仲間の割合）
kl = float((tg * (tg.clamp(min=1e-9).log2() - pb.clamp(min=1e-9).log2())).sum(1)[has].mean())   # 表との違い（KL、ビット）
print("学習", a.steps, "回 表との違い", round(kl, 3), "ビット", f"{time.time() - t0:.0f} 秒", flush=True)
torch.save(dict(P=Pn, words=WORDS, cluster=lab, k=a.k, table=tg, board=pb, source=a.source), f"{PRIV}/models/goban_next_{a.source}.pt")
with open("results/exp77.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(exp="77", source=a.source, k=a.k, steps=a.steps, kl_bits=kl, words_with_counts=int(has.sum()),
                            cluster_min=int(sizes.min()), cluster_max=int(sizes.max()), sec=time.time() - t0, device=dev), ensure_ascii=False) + "\n")
