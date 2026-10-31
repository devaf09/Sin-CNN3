"""実験80: 属性板 — 語板の先にもう 1 段の板を置いて、値の席を 2,000 に増やす（持ち主の案 2、2026-10-31）

着想: 持ち主「属性の関連にもうワンクッション、語板のようなグレイン（属性板）を置き、実質的な属性の数を増やしつつ、現実的な時間で学習する。
      荒い走査は 語板 → 属性、詳しい走査は 語板 → 属性 → 属性板」。docs/plan_large_scale.md 9.2 節。
問題: 観点「を」（何を〜する）の値を、上位 45（席に入る割合 0.41）から上位 2,000（0.99）に増やして当てる。名詞 6000 語（語板 v3 の 3 万語の 5 語おき）。
方法:
  値 2,000 を、名詞との組み合わせの似かたで 45 の仲間（荒い属性）に分ける（k 平均法、1 つの仲間は 240 まで）。値 = (仲間 c, 仲間の中の番号 m)。
  A 大きな器 1 枚: 語板 縦 2048 × 板 16、出口 2,000 + その他（比べの元）
  B 属性板（明るさだけ）: 語板 縦 128 × 板 16 が 仲間 45 + その他 の割合を出す。属性板 45 枚（縦 256 × 板 16、全部の語で共通）は、
     決まった波を入れて 仲間の中の割合 P(m | c) を出す。P(値 | 語) = P(c | 語) × P(m | c)。属性板は語によらない。
  C 属性板（位相込みの光を入れる）: 語板の出口の複素の光（縦 128 の全部の周波数）を、そのまま仲間 c の属性板の入口に入れる。
     属性板は共通の読み手で、語板は光の形に語ごとの細かい情報を書く。P(値 | 語) = P(c | 語) × P(m | c, 語の光)。語板と属性板を一緒に学ぶ。
  D 数えた表（平らにした、値 2,000）と、値 45 の表（いまの語板の形。席の外は「その他」）。
  当たり: 確かめ用の材料の交差エントロピー（ビット、値 2,000 + その他 の 2,001 通りで測る）。計算時間も記録。
結果: results/exp80.txt
実行: python experiments/exp80_attribute_boards.py --device cuda --method A（experiments/sweeps/sweep80_0.sh・1.sh）
"""
import argparse, collections, json, math, os, pickle, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--method", default="C")                                                  # A / B / C / D
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--nval", type=int, default=2000)
ap.add_argument("--ncl", type=int, default=45)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate, init_plates, _h, VAL0
import goban_batch

sys.stdout.reconfigure(encoding="utf-8"); t00 = time.time(); dev = a.device; PRIV = "D:/AI_git/Sin-CNN3_private"
VIEW = "を"
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
D = pickle.load(open(f"{PRIV}/data/views/merged2.pkl", "rb"))
ALL = [n for n in sorted(D["noun_total"], key=D["noun_total"].get, reverse=True) if n not in SKIP][:30000]
WORDS = ALL[::a.every]
tot = collections.Counter()
for n in ALL:
    tot.update(D["train"][n].get(VIEW, {}))
VALS = [x for x, _ in tot.most_common(a.nval)]; VI = {x: i for i, x in enumerate(VALS)}
NV = len(VALS)
T = {}
for s in ("train", "test"):
    X = torch.zeros(len(WORDS), NV + 1)                                                       # 最後 = その他（上位 2,000 の外）
    for i, n in enumerate(WORDS):
        for x, m in D[s].get(n, {}).get(VIEW, {}).items():
            X[i, VI.get(x, NV)] += m
    T[s] = X
del D
TR, TE = T["train"], T["test"]
print("表", len(WORDS), "語 × 値", NV, f"{time.time() - t00:.0f} 秒", flush=True)
pri = (TR.sum(0) + 0.5) / (TR.sum(0) + 0.5).sum()
tg = (TR + 20 * pri) / (TR.sum(1, keepdim=True) + 20)                                       # 平らにした表（値 2,000 + その他）
res = dict(exp="80", method=a.method, words=len(WORDS), nval=NV, ncl=a.ncl, steps=a.steps, device=dev)

# ---- 値の仲間分け（名詞との組み合わせの似かた。1 つの仲間は 240 まで） ----
Xv = torch.nn.functional.normalize(TR[:, :NV].T.sqrt(), dim=1)                               # 値ごとの「どの名詞と組むか」（平方根でならす）
g = torch.Generator().manual_seed(0); C = Xv[torch.randperm(NV, generator=g)[:a.ncl]].clone(); CAP = 240
for it in range(25):
    sim = Xv @ C.T; lab = torch.full((NV,), -1, dtype=torch.long); cnt = torch.zeros(a.ncl, dtype=torch.long)
    for v in torch.argsort(sim.max(1).values, descending=True).tolist():                     # 近い順に、空きのある仲間へ（大きさの上限つき）
        for c in torch.argsort(sim[v], descending=True).tolist():
            if cnt[c] < CAP:
                lab[v] = c; cnt[c] += 1; break
    for c in range(a.ncl):
        if (lab == c).any():
            C[c] = torch.nn.functional.normalize(Xv[lab == c].mean(0), dim=0)
MEM = [torch.nonzero(lab == c).flatten() for c in range(a.ncl)]                              # 仲間 c の値の番号
SZ = max(len(m) for m in MEM)
print("仲間分け", a.ncl, "組（大きさ", min(len(m) for m in MEM), "〜", SZ, "）", f"{time.time() - t00:.0f} 秒", flush=True)
res["cluster_examples"] = {c: [VALS[int(v)] for v in MEM[c][:8]] for c in range(0, a.ncl, 9)}


def coarse(X):
    """(語, 値 + その他) → (語, 仲間 + その他)。"""
    O = torch.zeros(X.shape[0], a.ncl + 1)
    for c in range(a.ncl):
        O[:, c] = X[:, MEM[c]].sum(1)
    O[:, a.ncl] = X[:, NV]
    return O


def fine_from(pc, pm):
    """P(c | 語)（語, 仲間 + その他）と P(m | c, 語)（list: 仲間ごとに (語, 大きさ)）→ P(値 | 語)（語, 値 + その他）。"""
    out = torch.zeros(pc.shape[0], NV + 1)
    for c in range(a.ncl):
        out[:, MEM[c]] = pc[:, c:c + 1] * pm[c]
    out[:, NV] = pc[:, a.ncl]
    return out / out.sum(1, keepdim=True)


def propagate(u, P):
    """入口の光 u（語, 縦）を板 P（語, 縦, 板）に通し、出口の複素の光（語, 縦、周波数ごと）を返す（goban_batch.run と同じ回折）。"""
    n = P.shape[1]; H = _h(u.device, n)
    u = torch.nn.functional.pad(u, (0, n))
    for k in range(P.shape[2]):
        u = torch.cat([u[..., :n] * torch.exp(1j * P[..., k]), u[..., n:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
    return torch.fft.fft(u[..., :n], dim=-1, norm="ortho")


t1 = time.time()
if a.method == "D":
    res["bits"] = evaluate(tg[:, None], TE[:, None])["bits"]
    rest = pri[45:] / pri[45:].sum()                                                        # 値 45 の表（いまの語板の形）: 上位 45 の外は、全体の割合で配る
    t45b = torch.cat([tg[:, :45], tg[:, 45:].sum(1, keepdim=True) * rest[None]], 1)
    res["bits_45table"] = evaluate(t45b[:, None], TE[:, None])["bits"]
elif a.method == "A":
    B = 300
    P = torch.cat([train(tg[i:i + B, None], torch.ones(len(tg[i:i + B]), 1, dtype=torch.bool), a.steps, i, lr=0.03, dev=dev, rows=2048, plates=16).cpu() for i in range(0, len(WORDS), B)])
    W = torch.cat([run(P[i:i + B].to(dev), 1, NV + 1).cpu() for i in range(0, len(WORDS), B)])[:, 0]
    res["bits"] = evaluate(prob(W)[:, None], TE[:, None])["bits"]
else:
    tgc = coarse(tg)                                                                         # 荒い目標 P(c | 語)
    Pw = init_plates(len(WORDS), 0, dev, 128, 16).requires_grad_(True)                      # 語板（縦 128 × 板 16）
    Pa = init_plates(a.ncl, 1, dev, 256, 16).requires_grad_(True)                           # 属性板 45 枚（縦 256 × 板 16）
    opt = torch.optim.Adam([Pw, Pa], lr=0.03)
    u0 = goban_batch._views(1, 128, dev)[0]                                               # 語板の入口 = 観点の波
    ua = goban_batch._views(1, 256, dev)[0]                                               # 属性板の入口（B: 決まった波）
    tgd = tg.to(dev); tgcd = tgc.to(dev); NB = 1500
    # 仲間ごとの細かい目標: 語ごとの P(m | c, 語)（その語がその仲間に光を持つときだけ学ぶ）
    HASM = (coarse(TR)[:, :a.ncl] > 0).to(dev)                                             # (語, 仲間) その語がその仲間に光を持つか

    def word_out(idx):
        """語板 idx の出口の複素の光（語, 縦 128）。"""
        return propagate(u0[None].expand(len(idx), -1), Pw[idx])

    def attr_out(c, f):
        """仲間 c の属性板に光 f（語, 縦 128 の複素）を入れ、仲間の中の割合 (語, 大きさ) を返す。B は決まった波を入れる。"""
        if a.method == "B":
            uin = ua[None].expand(f.shape[0], -1)
        else:
            uin = torch.nn.functional.pad(f / f.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-9), (0, 128))   # 語の光の形（強さはそろえる）を入口に
        o = propagate(uin, Pa[c][None].expand(f.shape[0], -1, -1)).abs()[..., VAL0:VAL0 + len(MEM[c])] ** 2
        return o / o.sum(-1, keepdim=True).clamp(min=1e-12)

    for step in range(a.steps):
        perm = torch.randperm(len(WORDS))
        for b0 in range(0, len(WORDS), NB):
            idx = perm[b0:b0 + NB].to(dev); f = word_out(idx)
            Wc = f.abs()[..., VAL0:VAL0 + a.ncl + 1] ** 2                                   # 荒い出口（仲間 45 + その他）
            pc = torch.cat([Wc, (1 - Wc.sum(-1, keepdim=True)).clamp(min=1e-9)], -1)
            loss = -(torch.cat([tgcd[idx], torch.zeros(len(idx), 1, device=dev)], -1) * torch.log(pc + 1e-9)).sum(-1).mean()
            for c in range(a.ncl):
                rows = HASM[idx, c].nonzero().flatten()
                if len(rows) == 0:
                    continue
                q = tgd[idx[rows]][:, MEM[c].to(dev)]; q = q / q.sum(-1, keepdim=True)
                pm = attr_out(c, f[rows])
                loss = loss - (tgcd[idx[rows], c:c + 1] * (q * torch.log(pm + 1e-9)).sum(-1, keepdim=True)).sum() / len(idx)   # 仲間の重みつきの細かい損失
            opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(a.method, step, "回", round(float(loss), 4), f"{time.time() - t00:.0f} 秒", flush=True)
    with torch.no_grad():
        pcs = []; pms = [[] for _ in range(a.ncl)]
        for b0 in range(0, len(WORDS), NB):
            idx = torch.arange(b0, min(b0 + NB, len(WORDS)), device=dev); f = word_out(idx)
            pcs.append(prob(f.abs()[..., VAL0:VAL0 + a.ncl + 1] ** 2).cpu())
            for c in range(a.ncl):
                pms[c].append(attr_out(c, f).cpu())
        pc = torch.cat(pcs); pm = [(0.98 * torch.cat(x) + 0.02 / len(MEM[c])) for c, x in enumerate(pms)]
        res["bits"] = evaluate(fine_from(pc, pm)[:, None], TE[:, None])["bits"]
        res["bits_coarse"] = evaluate(pc[:, None], coarse(TE)[:, None])["bits"]
        res["table_coarse"] = evaluate(tgc[:, None], coarse(TE)[:, None])["bits"]
res["table"] = evaluate(tg[:, None], TE[:, None])["bits"]; res["sec"] = time.time() - t1
with open("results/exp80.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print({k: v for k, v in res.items() if k != "cluster_examples"})
