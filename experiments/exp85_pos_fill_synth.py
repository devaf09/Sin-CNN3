"""実験85: 品詞の穴埋めシンセ — 句読点で挟まれた区切りの品詞の並びを、そのまま学ぶ（持ち主の案 2026-11-01）

着想: 持ち主「句読点で挟まれた文法の品詞を、そのまま学習するシンセを作ってくれる？今までは前後の品詞の流れを学習する仕組みだったけど、
      このシンセは今の文から今の文の品詞を出力する形で。試験の一つに、文章の品詞を抜いて、それでも確率的に品詞を穴埋めできるかやってほしい」。
      あわせて実験84 の直し（器の学び方、当てるものを「行の中の次の品詞」に）: 次の品詞 = 最後の語を抜いた穴埋め、として同じ器で測る。
材料: 実験84 と同じ（会話・青空文庫 300 作品・Web 小説 3000 話、実験だけ、数だけ）の、句読点（、。！？）で区切った単位。語の数 2〜24。
入力の光: 位置 t の語の品詞 k = 入口の周波数 k + 1、位置 = 位相（2π t / 25）。抜いた所は「穴」の周波数（品詞の数 + 1）を同じ位相で入れる（穴の位置は分かる）。
出力: 位置ごとに品詞 80 通りの出口（24 位置 × 80 = 1,920 の出口）。器 = 縦 2048 × 板 16（全部の文で共通の 1 つ）。
学び方（実験84 の直し）: 回数 6000 回、学習率を途中で下げる、抜く割合を 0〜60% でばらばらにして、全部の位置の品詞を出させる（今の文 → 今の文 + 穴埋め）。
試験:
  (a) 穴埋め: 確かめ用の材料で、15%・30%・50% の語の品詞を抜き、抜いた所の品詞を当てる（上位 1 の当たり・交差エントロピー）。
  (b) 次の品詞: 区切りの最後の語の品詞を抜いて当てる（ここまでの品詞から、次の品詞）。
  (c) そのまま（抜かない）: 今の文の品詞をそのまま出せるか。
比べる相手（光を使わない数え方）: 位置ごとの品詞の割合 / 両どなりの品詞から当てる表（P(品詞 | 左, 右)、となりも抜かれていれば片側・全体の割合へ下げる）。
結果: results/exp85.txt。実行: python experiments/exp85_pos_fill_synth.py --device cuda
"""
import argparse, collections, json, math, os, pickle, random, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--rows", type=int, default=2048)
ap.add_argument("--plates", type=int, default=16)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--enc", default="phase")                                                  # phase: 位置 = 位相 / freq: 位置と品詞の組ごとに別の周波数（1 + 品詞 + 81 × 位置）
ap.add_argument("--ref", action="store_true")
ap.add_argument("--kerr", type=float, default=0.0)                                         # 板の真ん中でカー効果（明るいほど位相が遅れる、実験54）。となりとの組み合わせを覚えやすくする試み                                              # いつも点いている光（参照の光）を足す。位相は参照がないと読めない（2026-11-01 の気づき）
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import init_plates, _h, VAL0

sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time(); dev = a.device; PRIV = "D:/AI_git/Sin-CNN3_private"
DOCS = pickle.load(open(f"{PRIV}/data/views/synth_units.pkl", "rb"))                      # 実験84 で作った行ごとの (語, 品詞の値) の並び
PUNCT = {"、", "。", "！", "？", "!", "?", "，", "．"}; L = 24
U = {"train": [], "test": []}
for s, rows in DOCS:
    for r in rows:
        cur = []
        for w, l in r:
            cur.append(l)
            if w in PUNCT:
                if 2 <= len(cur) <= L:
                    U[s].append(cur)
                cur = []
        if 2 <= len(cur) <= L:
            U[s].append(cur)
del DOCS
lc = collections.Counter(l for u in U["train"] for l in u)
LABS = [l for l, _ in lc.most_common(80)]; LI = {l: i for i, l in enumerate(LABS)}; NL = len(LABS); HOLE = NL   # 品詞 80（外は最後の番号）、穴 = 80
random.Random(0).shuffle(U["train"]); U["train"] = U["train"][:(20000 if a.quick else 600000)]; U["test"] = U["test"][:(2000 if a.quick else 30000)]


def tens(us):
    X = torch.full((len(us), L), -1, dtype=torch.long)
    for i, u in enumerate(us):
        X[i, :len(u)] = torch.tensor([LI.get(l, NL - 1) for l in u])
    return X


Xtr, Xte = tens(U["train"]), tens(U["test"])
print("区切り", len(Xtr), len(Xte), "品詞", NL, f"{time.time() - t0:.0f} 秒", flush=True)
g = torch.Generator().manual_seed(1)


def mask(X, rate, last=False):
    """抜く所の印（語のある所だけ）。last = 最後の語だけ抜く（次の品詞）。"""
    has = X >= 0
    if last:
        M = torch.zeros_like(has); n = has.sum(1); M[torch.arange(len(X)), NEXT_T] = True
        return M & has
    return (torch.rand(X.shape, generator=g) < rate) & has


# ---- 比べる相手（光を使わない） ----
pos_prior = torch.zeros(L, NL)
for t in range(L):
    v = Xtr[:, t]; pos_prior[t] = torch.bincount(v[v >= 0], minlength=NL).float() + 0.5
pos_prior /= pos_prior.sum(1, keepdim=True)
LR = collections.defaultdict(collections.Counter); Lft = collections.defaultdict(collections.Counter); Rgt = collections.defaultdict(collections.Counter)
for row in Xtr.tolist():
    row = [x for x in row if x >= 0]; ext = [NL + 1] + row + [NL + 2]                          # 区切りの始め・終わりも「となり」として使う
    for t in range(1, len(ext) - 1):
        LR[(ext[t - 1], ext[t + 1])][ext[t]] += 1; Lft[ext[t - 1]][ext[t]] += 1; Rgt[ext[t + 1]][ext[t]] += 1
glob = torch.bincount(Xtr[Xtr >= 0], minlength=NL).float() + 0.5; glob /= glob.sum()


def table_pred(X, M, open_end=False):
    """両どなりの品詞（抜かれていなければ）から当てる表。両方分かれば P(・| 左, 右)、片方なら片側、なければ全体の割合。"""
    out = torch.zeros(X.shape[0], L, NL)
    for i in range(X.shape[0]):
        row = X[i].tolist(); n = sum(1 for x in row if x >= 0)
        for t in range(n):
            if not M[i, t]:
                continue
            lft = (NL + 1) if t == 0 else (row[t - 1] if not M[i, t - 1] else None)
            rgt = (None if open_end else NL + 2) if t == n - 1 else (row[t + 1] if not M[i, t + 1] else None)
            c = LR.get((lft, rgt)) if lft is not None and rgt is not None else None
            if not c:
                c = Lft.get(lft) if lft is not None else (Rgt.get(rgt) if rgt is not None else None)
            p = glob.clone() * 0.5
            if c:
                for k, m in c.items():
                    if k < NL:
                        p[k] += m
            out[i, t] = p / p.sum()
    return out


def score(P, X, M):
    """抜いた所の上位 1 の当たりと、交差エントロピー（ビット）。"""
    t = X[M]; p = P[M]
    return dict(acc=float((p.argmax(-1) == t).float().mean()), bits=float(-torch.log2(p.gather(-1, t[:, None]).clamp(min=1e-9)).mean()), n=int(M.sum()))


# ---- 光の器（文法シンセ） ----
N, K = a.rows, a.plates; H = _h(dev, N)
yy = torch.arange(N, dtype=torch.float32, device=dev)
NF = (NL + 1) * L if a.enc == "freq" else NL + 1                                            # 入口の周波数の数（freq なら 位置 × (品詞 80 + 穴)）
WAVE = torch.stack([torch.exp(1j * 2 * math.pi * (k + 1) * yy / N) / math.sqrt(N) for k in range(NF + 1)]).to(torch.complex64)   # 最後 = 参照の光
PH = torch.exp(1j * 2 * math.pi * torch.arange(L, device=dev) / 25).to(torch.complex64)


def encode(X, M):
    X = X.to(dev); M = M.to(dev); has = (X >= 0)
    idx = torch.where(M, torch.full_like(X, HOLE), X.clamp(min=0))
    if a.enc == "freq":
        idx = idx + (NL + 1) * torch.arange(L, device=dev)[None]                             # 位置ごとに別の周波数の組
        u = torch.einsum("bt,btn->bn", has.to(torch.complex64), WAVE[idx])
    else:
        u = torch.einsum("bt,btn->bn", has.to(torch.complex64) * PH[None], WAVE[idx])
    u = u / u.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-9)
    if a.ref:
        u = (u + WAVE[NF][None]) / math.sqrt(2)                                              # 参照の光（位相を読むための、いつも点いている光）
    return u


def forward(P, u):
    u = torch.nn.functional.pad(u, (0, N))
    for k in range(K):
        u = torch.cat([u[..., :N] * torch.exp(1j * P[:, k]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
        if a.kerr and k == K // 2 - 1:                                                       # 真ん中の焦点面（周波数の面）でカー効果: 明るさ ÷ 平均 に比例して位相が遅れる
            f = torch.fft.fft(u, dim=-1, norm="ortho"); I0 = f.abs() ** 2
            f = f * torch.exp(1j * a.kerr * I0 / I0.mean(-1, keepdim=True).clamp(min=1e-12))
            u = torch.fft.ifft(f, dim=-1, norm="ortho")
    I = torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs() ** 2
    out = I[:, VAL0:VAL0 + L * NL].reshape(-1, L, NL)
    return out / out.sum(-1, keepdim=True).clamp(min=1e-12)                                  # 位置ごとに割合へ


P = init_plates(1, 0, dev, N, K)[0].requires_grad_(True)
opt = torch.optim.Adam([P], lr=0.02); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
for st in range(a.steps):
    idx = torch.randint(0, len(Xtr), (a.batch,)); X = Xtr[idx]
    rate = torch.rand(a.batch, 1, generator=g) * 0.6                                         # 抜く割合 0〜60% をばらばらに
    M = (torch.rand(X.shape, generator=g) < rate) & (X >= 0)
    o = forward(P, encode(X, M)); has = (X >= 0).to(dev)
    lp = torch.log(o.gather(-1, X.clamp(min=0).to(dev)[..., None])[..., 0] + 1e-9)
    w = has.float() * (1 + 2 * M.to(dev).float())                                            # 全部の位置を出させ、抜いた所を 3 倍に重く
    loss = -(lp * w).sum() / w.sum()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if st % 500 == 0:
        print(st, "回", round(float(loss), 4), f"{time.time() - t0:.0f} 秒", flush=True)
res = dict(exp="85", enc=a.enc, ref=a.ref, kerr=a.kerr, units=len(Xtr), labels=NL, rows=N, plates=K, steps=a.steps)
# 次の品詞: 区切りの中の位置 t（1 以上）をでたらめに選び、t より後ろを消して（ここまでしか見えない）、t を抜く
NEXT_T = torch.tensor([int(torch.randint(1, max(2, int(n)), (1,), generator=g)) for n in (Xte >= 0).sum(1)])
Xnext = Xte.clone()
for i, t in enumerate(NEXT_T.tolist()):
    Xnext[i, t + 1:] = -1
for name, rate, last in (("穴埋め 15%", 0.15, False), ("穴埋め 30%", 0.30, False), ("穴埋め 50%", 0.50, False), ("次の品詞（ここまでの品詞から）", 0, True), ("そのまま", 0, False)):
    XX = Xnext if last else Xte
    M = mask(XX, rate, last)
    if name == "そのまま":
        M = Xte >= 0; Mi = torch.zeros_like(M)                                                # 抜かずに入れて、全部の位置を出させる
    else:
        Mi = M
    with torch.no_grad():
        O = torch.cat([forward(P, encode(XX[i:i + 1000], Mi[i:i + 1000])).cpu() for i in range(0, len(XX), 1000)])
    O = 0.99 * O + 0.01 / NL
    r = dict(光の器=score(O, XX, M), 位置ごとの割合=score(pos_prior[None].expand(len(XX), -1, -1), XX, M))
    if name != "そのまま":
        r["両どなりの表"] = score(table_pred(XX, M, open_end=last), XX, M)                                    # 次の品詞では右どなりが無いので、左どなりだけの表になる
    res[name] = r
    print(name, r, flush=True)
res["sec"] = time.time() - t0
torch.save(dict(P=P.detach().cpu(), labels=LABS, enc=a.enc, ref=a.ref), f"{PRIV}/models/synth_fill_{a.enc}{'_ref' if a.ref else ''}_k{K}{'_kerr' if a.kerr else ''}.pt")
with open("results/exp85.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
