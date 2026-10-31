"""実験73: 語板 v2 — 観点 9（どんな・を・が・で・に・の前・の後・と・は）、名詞 3 万語。リフレッシュ学習（v1 の板から学び直す）と、まっさらから学ぶのを比べる

着想: 2026-10-29 持ち主「属性樹と語板の充実が今後の実験に必要」「語板の学習は、追加された語が属性の取得が少ない可能性があるから、定期的にリフレッシュ学習をする。
      最初は語板を残して学習の後にもう一度学習するなど」。観点を増やして数え直した材料（実験72 と同じ merged2.pkl、会話 × 100、R15・R18 × 0.1）。
方法: 名詞 = 多い順に 3 万語（形式名詞を除く）。値 = 観点ごとに全体で多い順 45 個 + 観点ごとの「その他」（出口 45 + 9 = 54、縦 64 の器の上限）。出口のずらし（実験58、45 で回す）。
      目標 = 平らにした表（k = 20）。器 = 縦 64・板 8 枚（src/goban_batch.py）。2000 回・学習率 0.03。
      リフレッシュ学習: 実験59 の語板（v1、観点 5）がある語は、その板から学び直す（器は観点の数によらないので、板はそのまま使える）。ない語はまっさら。
      比べ: 全部まっさらから（--scratch）。学習の途中（250・500・1000・2000 回）の当たりも記録する。
      語を 2 つに分けて（--part 0 / 1）、GPU 2 枚で同時に。
結果: results/exp73.txt。語板: 非公開フォルダの models/goban_v2_part<k>[_scratch].pt
実行: python experiments/exp73_goban_v2.py --device cuda --part 0（experiments/sweeps/sweep73_p0.sh・p1.sh）
"""
import argparse, collections, json, math, os, pickle, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--part", type=int, default=0)
ap.add_argument("--parts", type=int, default=2)
ap.add_argument("--nouns", type=int, default=30000)
ap.add_argument("--device", default="cpu")
ap.add_argument("--scratch", action="store_true")
ap.add_argument("--out", default="results/exp73.txt")
ap.add_argument("--rows", type=int, default=64)                                           # 器の縦（語板 v3 = 128、実験75）
ap.add_argument("--plates", type=int, default=8)                                          # 板の枚数（語板 v3 = 16）
ap.add_argument("--steps", type=int, default=2000)                                        # 学習の回数（語板 v3 = 500 で十分、実験75）
ap.add_argument("--nv", type=int, default=45)                                             # 1 観点の値の席（語板 v4 = 490、縦 512 の器に入る数。持ち主 2026-10-31「席の数を増やした板」）
ap.add_argument("--name", default="v2")                                                   # 保存名 goban_<name>_part<k>.pt
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate as _evaluate


def evaluate(Pr, counts):
    """値が多いとき（--nv 490 など）は、上位 5 の順位を数えず、交差エントロピーと上位 1 だけを測る（goban_batch.evaluate は値の数の 2 乗の記憶を使う）。"""
    if Pr.shape[-1] <= 100:
        return _evaluate(Pr, counts)
    Pr = Pr.cpu().reshape(-1, Pr.shape[-1]); w = counts.reshape(-1, counts.shape[-1]); s_ = w.sum().clamp(min=1)
    return dict(bits=float(-(w * torch.log2(Pr.clamp(min=1e-9))).sum() / s_), top1=float(w.gather(1, Pr.argmax(1, keepdim=True)).sum() / s_), n=int(s_))

sys.stdout.reconfigure(encoding="utf-8"); t00 = time.time(); dev = a.device
VIEWS = ["どんな", "を", "が", "で", "に", "の前", "の後", "と", "は"]; NV = a.nv
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
D = pickle.load(open("D:/AI_git/Sin-CNN3_private/data/views/merged2.pkl", "rb"))
ALL = [n for n in sorted(D["noun_total"], key=D["noun_total"].get, reverse=True) if n not in SKIP][:a.nouns]
allv = {v: collections.Counter() for v in VIEWS}
for n in ALL:
    for v in VIEWS:
        allv[v].update(D["train"][n].get(v, {}))
VALS = {v: [x for x, _ in allv[v].most_common(NV)] for v in VIEWS}
POS = {v: {x: (r + 9 * j) % NV for r, x in enumerate(VALS[v])} for j, v in enumerate(VIEWS)}
WORDS = ALL[a.part::a.parts]                                                              # この仕事で学ぶ語（2 つに分けた片方）


def table(split):
    T = torch.zeros(len(WORDS), len(VIEWS), NV + len(VIEWS))
    for i, n in enumerate(WORDS):
        dd = D[split].get(n, {})
        for j, v in enumerate(VIEWS):
            for x, m in dd.get(v, {}).items():
                T[i, j, POS[v].get(x, NV + j)] += m
    return T


TR, TE = table("train"), table("test"); del D
pri = (TR.sum(0) + 0.5) / (TR.sum(0) + 0.5).sum(1, keepdim=True)
tg = (TR + 20 * pri[None]) / (TR.sum(2, keepdim=True) + 20)
print("表", len(WORDS), "語", f"{time.time() - t00:.0f} 秒", flush=True)
# 初期の板: v1（実験59）の板があればそこから（リフレッシュ）、なければまっさら（2 値の板）
from goban_batch import init_plates
P0 = init_plates(len(WORDS), seed=a.part, rows=a.rows, plates=a.plates)
n_v1 = 0
if not a.scratch and a.rows == 64 and a.plates == 8:                                       # v1 の板が使えるのは同じ大きさの器だけ
    v1 = torch.load("D:/AI_git/Sin-CNN3_private/models/goban_名詞_seed0.pt", weights_only=False)
    vi = {w: i for i, w in enumerate(v1["words"])}
    for i, n in enumerate(WORDS):
        if n in vi:
            P0[i] = v1["P"][vi[n]]; n_v1 += 1
rows = []; P = P0
B = 5000 if a.rows * a.plates <= 512 else 1500 if a.rows <= 128 else 300; done = 0                               # 大きな器は GPU のメモリに合わせて小分け
for steps in {500: (250, 250), 1000: (250, 250, 500)}.get(a.steps, (250, 250, 500, 1000)):          # 250・500・1000・2000 回の時点で当たりを測る（--steps まで）
    P = torch.cat([train(tg[i:i + B], torch.ones(tg[i:i + B].shape[:2], dtype=torch.bool), steps, a.part + i + done, lr=0.03, P0=P[i:i + B], dev=dev).cpu() for i in range(0, len(WORDS), B)])
    done += steps
    W = torch.cat([run(P[i:i + B].to(dev), len(VIEWS), NV + len(VIEWS)).cpu() for i in range(0, len(WORDS), B)])
    ev = evaluate(prob(W), TE)
    rows.append(dict(steps=done, bits=ev["bits"], light=float(W.sum(-1).mean())))
    print(f"{done} 回: 当たり {ev['bits']:.3f}  光 {rows[-1]['light']:.3f}  {time.time() - t00:.0f} 秒", flush=True)
res = dict(exp="73", name=a.name, rows=a.rows, plates=a.plates, part=a.part, words=len(WORDS), from_v1=n_v1, scratch=a.scratch, curve=rows, table=evaluate(tg, TE)["bits"],
           no_word=evaluate(pri[None].expand(len(WORDS), -1, -1), TE)["bits"], sec=time.time() - t00, device=dev)
torch.save(dict(P=P, words=WORDS, views=VIEWS, vals=VALS, pos=POS), f"D:/AI_git/Sin-CNN3_private/models/goban_{a.name}_part{a.part}{'_scratch' if a.scratch else ''}.pt")
with open(a.out, "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print("表", round(res["table"], 3), "語を見ない", round(res["no_word"], 3), "v1 から", n_v1)
