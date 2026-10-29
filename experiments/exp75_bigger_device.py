"""実験75: 器を大きくする — 語板 v2（観点 9）の「表との差 0.2 ビット」は、縦・板の数を増やすと縮むか。値を増やしても学べるか

着想: 2026-10-30 実験73 で、観点 5 → 9 にすると語板と表の差が 0.03〜0.05 → 0.2 ビットに広がった（器 = 縦 64・板 8 枚）。
      持ち主「今後も情報は増えるから、先に器を大きくする試験をしてもいい」。
方法: 材料・目標は実験73 と同じ（merged2.pkl、観点 9、平らにした表 k = 20、出口のずらし）。名詞 3 万語から 5 語おきに 6000 語（多い語もまれな語も入る）。
      器: 縦 64・板 8（いま）/ 縦 64・板 16 / 縦 128・板 8 / 縦 128・板 16。列の間（回折の距離）はどれも 64。
      値を増やす: 縦 128 なら出口に 100 値 + その他 も置ける（周波数 10〜118）。値 45 と 100 で、それぞれ自分の表との差を見る。
      2000 回・学習率 0.03。250・500・1000・2000 回の時点の当たり（確かめ用の材料の交差エントロピー、ビット）を記録。
      --job 0: 縦 64 の 2 つと縦 128・板 8、--job 1: 縦 128・板 16（値 45 と 100）。GPU 2 枚で同時に。
結果: results/exp75.txt
実行: python experiments/exp75_bigger_device.py --device cuda --job 0（experiments/sweeps/sweep75_0.sh・1.sh）
"""
import argparse, collections, json, os, pickle, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--job", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--every", type=int, default=5)                                           # 名詞 3 万語から何語おきに取るか
ap.add_argument("--out", default="results/exp75.txt")
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

sys.stdout.reconfigure(encoding="utf-8"); t00 = time.time(); dev = a.device
VIEWS = ["どんな", "を", "が", "で", "に", "の前", "の後", "と", "は"]
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
D = pickle.load(open("D:/AI_git/Sin-CNN3_private/data/views/merged2.pkl", "rb"))
ALL = [n for n in sorted(D["noun_total"], key=D["noun_total"].get, reverse=True) if n not in SKIP][:30000]   # 実験73 と同じ 3 万語
WORDS = ALL[::a.every]                                                                      # その 5 語おき（多い語もまれな語も入る）
allv = {v: collections.Counter() for v in VIEWS}
for n in ALL:
    for v in VIEWS:
        allv[v].update(D["train"][n].get(v, {}))


def tables(nv):
    """値 nv 個 + 観点ごとの「その他」の表（学習用・確かめ用）と、平らにした目標。出口のずらし（実験58、nv で回す）。"""
    vals = {v: [x for x, _ in allv[v].most_common(nv)] for v in VIEWS}
    pos = {v: {x: (r + 9 * j) % nv for r, x in enumerate(vals[v])} for j, v in enumerate(VIEWS)}
    T = {}
    for split in ("train", "test"):
        T[split] = torch.zeros(len(WORDS), len(VIEWS), nv + len(VIEWS))
        for i, n in enumerate(WORDS):
            dd = D[split].get(n, {})
            for j, v in enumerate(VIEWS):
                for x, m in dd.get(v, {}).items():
                    T[split][i, j, pos[v].get(x, nv + j)] += m                              # 上位 nv 値の外は「その他」へ
    TR, TE = T["train"], T["test"]
    pri = (TR.sum(0) + 0.5) / (TR.sum(0) + 0.5).sum(1, keepdim=True)                       # 観点ごとの全体の割合
    tg = (TR + 20 * pri[None]) / (TR.sum(2, keepdim=True) + 20)                             # 平らにした目標（実験57）
    return tg, TE, pri


CONF = [(64, 8, 45), (64, 16, 45), (128, 8, 45)] if a.job == 0 else [(128, 16, 45), (128, 16, 100)]   # (縦, 板, 値の数)
cache = {}
for rows, plates, nv in CONF:
    if nv not in cache:
        cache[nv] = tables(nv); print("表 値", nv, len(WORDS), "語", f"{time.time() - t00:.0f} 秒", flush=True)
    tg, TE, pri = cache[nv]; nx = nv + len(VIEWS)
    B = 3000 if rows * plates <= 1024 else 1500                                             # 大きな器は GPU のメモリに合わせて小分け
    P = None; done = 0; curve = []; t1 = time.time()
    for steps in (250, 250, 500, a.steps - 1000):                                           # 250・500・1000・2000 回の時点で測る
        P = torch.cat([train(tg[i:i + B], torch.ones(tg[i:i + B].shape[:2], dtype=torch.bool), steps, i + done, lr=0.03,
                             P0=None if P is None else P[i:i + B], dev=dev, rows=rows, plates=plates).cpu() for i in range(0, len(WORDS), B)])
        done += steps
        W = torch.cat([run(P[i:i + B].to(dev), len(VIEWS), nx).cpu() for i in range(0, len(WORDS), B)])
        ev = evaluate(prob(W), TE); curve.append(dict(steps=done, bits=ev["bits"], top1=ev["top1"], light=float(W.sum(-1).mean())))
        print(rows, plates, nv, done, "回", round(ev["bits"], 3), f"{time.time() - t00:.0f} 秒", flush=True)
    res = dict(exp="75", rows=rows, plates=plates, vals=nv, words=len(WORDS), curve=curve, table=evaluate(tg, TE)["bits"],
               no_word=evaluate(pri[None].expand(len(WORDS), -1, -1), TE)["bits"], train_sec=time.time() - t1, device=dev)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print("表", round(res["table"], 3), "語を見ない", round(res["no_word"], 3), flush=True)
