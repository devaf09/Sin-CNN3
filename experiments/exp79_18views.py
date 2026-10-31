"""実験79: 観点を 18 に増やしたとき、器（縦 128 × 板 16・24・32）で当たりが保てるか

着想: 2026-10-31 持ち主「今の観点周波数ってどのくらいサポート出来てて、語板で最大どのくらい使ってるか」「次の試験も進めて OK」。
      語板 v3 の観点 9（どんな・を・が・で・に・の前・の後・と・は）に、品詞の観点 8（実験78 の D 案）と人称の観点 1 を足すと 18 観点。
方法: 語 = 語板 v3 の名詞 3 万語から 5 語おきに 6000 語（実験75 と同じ選び方）。
  観点 1〜9: 語板 v3 と同じ材料（merged2.pkl）・値 45 + その他・出口のずらし（実験58）。
  観点 10〜17: 品詞（実験78 の語板 goban_pos.pt の数えた割合を目標に。大分類 14・中分類）。
  観点 18: 人称（一人称・二人称・三人称・人称なし + その他。docs/grammar.md 2 節の一覧で決める。名詞なので、ほとんどが人称なし）。
  出口は観点のまとまりごとに場所を分ける（語板 v3 の観点 = 出口 0〜53、品詞 = 54〜75、人称 = 76〜81。違うまとまりの光が同じ出口に集まらない）。
  器: 観点 9 だけ・板 16（比べの元）/ 観点 18・板 16 / 24 / 32。500 回。
  測るもの: 観点 1〜9 の当たり（確かめ用の材料の交差エントロピー、ビット、表との差）。品詞の観点の、目標との違い（KL、ビット）と大分類の上位 1 の一致。計算時間。
結果: results/exp79.txt
実行: python experiments/exp79_18views.py --device cuda --job 0（experiments/sweeps/sweep79_0.sh・1.sh）
"""
import argparse, collections, json, os, pickle, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--job", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--every", type=int, default=5)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

sys.stdout.reconfigure(encoding="utf-8"); t00 = time.time(); dev = a.device; PRIV = "D:/AI_git/Sin-CNN3_private"
VIEWS = ["どんな", "を", "が", "で", "に", "の前", "の後", "と", "は"]; NV = 45
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
D = pickle.load(open(f"{PRIV}/data/views/merged2.pkl", "rb"))
ALL = [n for n in sorted(D["noun_total"], key=D["noun_total"].get, reverse=True) if n not in SKIP][:30000]
WORDS = ALL[::a.every]
allv = {v: collections.Counter() for v in VIEWS}
for n in ALL:
    for v in VIEWS:
        allv[v].update(D["train"][n].get(v, {}))
VALS = {v: [x for x, _ in allv[v].most_common(NV)] for v in VIEWS}
POS = {v: {x: (r + 9 * j) % NV for r, x in enumerate(VALS[v])} for j, v in enumerate(VIEWS)}
# ---- 観点 1〜9（語板 v3 と同じ） ----
TR = {}
for split in ("train", "test"):
    T = torch.zeros(len(WORDS), 9, NV + 9)
    for i, n in enumerate(WORDS):
        dd = D[split].get(n, {})
        for j, v in enumerate(VIEWS):
            for x, m in dd.get(v, {}).items():
                T[i, j, POS[v].get(x, NV + j)] += m
    TR[split] = T
del D
pri = (TR["train"].sum(0) + 0.5) / (TR["train"].sum(0) + 0.5).sum(1, keepdim=True)
tg1 = (TR["train"] + 20 * pri[None]) / (TR["train"].sum(2, keepdim=True) + 20)               # 平らにした表（k = 20）
# ---- 観点 10〜17（品詞、実験78 の数えた割合） ----
PM = torch.load(f"{PRIV}/models/goban_pos.pt", weights_only=False); PWI = {w: i for i, w in enumerate(PM["words"])}
NPV = len(PM["views"]); NPO = PM["nout"]; PHYS = PM["phys"]
tg2 = torch.zeros(len(WORDS), NPV, NPO); use2 = torch.zeros(len(WORDS), NPV, dtype=torch.bool)
for i, n in enumerate(WORDS):
    if n in PWI:
        t = PM["table"][PWI[n]]                                                              # (観点, 値 + その他) の割合
        for j in range(NPV):
            tg2[i, j, PHYS[j]] = t[j]
        use2[i] = PM["table"][PWI[n]].sum(-1) > 0
        use2[i, 1:] &= tg2[i, 1:].sum(-1) > 0
# 使う観点 = 数えた割合が「その他だけ」でない観点（その品詞にならない語は使わない）
for i in range(len(WORDS)):
    for j in range(1, NPV):
        if float(tg2[i, j, PHYS[j][:-1]].sum()) < 0.5:
            use2[i, j] = False
# ---- 観点 18（人称） ----
P1 = set("私 俺 オレ 僕 我 わし 余 我が輩 拙者 某 わっし わて オラ 己等 オイラ アタイ こちとら うち うら 拙僧 身共 朕 麻呂 乃公 わちき わっち 吾人 妾 余輩 私達 我々 僕達 自分 小生 当方".split())
P2 = set("貴方 御前 君 きみ 貴様 汝 お主 てまえ 貴公 貴殿 御宅 其の方 貴下 其処許 貴君 御辺 貴兄 うぬ ウヌ わりゃあ 貴辺 雅兄 君達".split())
P3 = set("彼 彼女 彼奴 此奴 其奴 此の方 どいつ どなた 誰".split())
tg3 = torch.zeros(len(WORDS), 1, 6)
for i, n in enumerate(WORDS):
    tg3[i, 0, 0 if n in P1 else 1 if n in P2 else 2 if n in P3 else 3] = 1.0                 # 一人称・二人称・三人称・人称なし（4 番目 = その他、使わない）
tg3 = (tg3 + 0.01) / (1 + 0.06)
# ---- 出口をまとまりごとに分けてつなぐ ----
O1, O2, O3 = NV + 9, NPO, 6; NOUT = O1 + O2 + O3                                            # 54 + 22 + 6 = 82（縦 128 なら 118 まで置ける）


def full(n18):
    """観点 9 / 18 の目標と、使う印。"""
    if not n18:
        return tg1, torch.ones(len(WORDS), 9, dtype=torch.bool)
    T = torch.zeros(len(WORDS), 18, NOUT)
    T[:, :9, :O1] = tg1; T[:, 9:17, O1:O1 + O2] = tg2; T[:, 17:, O1 + O2:] = tg3
    U = torch.cat([torch.ones(len(WORDS), 9, dtype=torch.bool), use2, torch.ones(len(WORDS), 1, dtype=torch.bool)], 1)
    return T, U


CONF = [(9, 16), (18, 16)] if a.job == 0 else [(18, 24), (18, 32)]
for nview, plates in CONF:
    T, U = full(nview == 18); B = 1500; t1 = time.time()
    P = torch.cat([train(T[i:i + B], U[i:i + B], a.steps, i, lr=0.03, dev=dev, rows=128, plates=plates).cpu() for i in range(0, len(WORDS), B)])
    if dev.startswith("cuda"):
        torch.cuda.synchronize()
    sec = time.time() - t1
    W = torch.cat([run(P[i:i + B].to(dev), nview, T.shape[2]).cpu() for i in range(0, len(WORDS), B)])
    r = dict(exp="79", views=nview, plates=plates, words=len(WORDS), steps=a.steps, train_sec=sec,
             bits_v3=evaluate(prob(W[:, :9, :O1]), TR["test"])["bits"], table_v3=evaluate(tg1, TR["test"])["bits"])
    if nview == 18:
        pp = prob(torch.stack([W[:, 9 + j, O1 + PHYS[j]] for j in range(NPV)], 1))            # 品詞の観点（値 + その他）
        tl = torch.stack([tg2[:, j, PHYS[j]] for j in range(NPV)], 1)
        kl = (tl * (tl.clamp(min=1e-9).log2() - pp.clamp(min=1e-9).log2())).sum(-1)
        r["pos_kl_bits"] = float(kl[use2].mean())
        has = use2[:, 0]
        r["pos_top1_agree"] = float((pp[has, 0, :14].argmax(1) == tl[has, 0, :14].argmax(1)).float().mean())
        p3 = prob(W[:, 17:, O1 + O2:]); r["person_agree"] = float((p3[:, 0].argmax(1) == tg3[:, 0].argmax(1)).float().mean())
    with open("results/exp79.txt", "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(r, f"{time.time() - t00:.0f} 秒", flush=True)
