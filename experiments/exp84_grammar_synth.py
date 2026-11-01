"""実験84: 文法シンセ — 1 行（または句読点まで）の品詞の並びから、次の行の品詞を当てる（持ち主の案 2026-11-01）

着想: 持ち主「品詞の All in All out 方式。1 行を取り出し、語板の属性（品詞）を取り出す（今日 は 晴れ まし た 。→ 名詞 助詞 動詞 助動詞 助動詞 。）。
      次の行も同じように取り出し、前の行を入力、次の行を出力の正解として学習する。これを文法シンセと呼ぶ」。
  方式 1: 品詞の並びをそのまま入れる / 2: 1 に畳み込み（前後の入れ替えに少し対応）/
  3: 観点周波数ごとに品詞のタグを付け、行の中で重要な語（動詞との結びつきが大きい語）の品詞を、重要度のグラデーションで入れる（足切り 50%）/ 4: 3 に畳み込み
  5〜8: 1〜4 を行ではなく句読点（、。！？）の区切りで。
器（文法シンセ）: 全部の文で共通の器 1 つ（縦 512 × 板 16、語板より幅広で深い）。たくさんの行の組で学ぶ。逆引きできる形はそのまま。
入力の光: 品詞 = 入口の周波数（品詞の番号 + 1）、位置 = 位相（位置 t の語は位相 2π t / 25 だけずらす）。並びの語の光を足し合わせる（重なりで並びを表す）。
  畳み込み: 位置 t の語を、t − 1・t・t + 1 の位相にも 0.5・1・0.5 の重みで広げる（となりと入れ替わっても光の形があまり変わらない）。
  重要度（3・4）: 動詞（一般）= 1、それ以外の語 = その語のあとの 3 語以内に動詞が来る割合（材料で数える = 動詞との結びつき）を、行の中のいちばん大きい値で割る。
               0.5 未満は 0（足切り）。句読点・助詞・助動詞は語そのものの重要度で、ふつうは低い。
出力（何を当てるか、ザックリ全体）: 次の行の品詞の割合（品詞ごとの数の割合）と、次の行の最初の 3 つの品詞。出口は「割合」80 + 「1 番目」80 + 「2 番目」80 + 「3 番目」80。
当たり: 確かめ用の材料で、割合の交差エントロピー（ビット）、最初の 3 つの上位 1 の当たり。
比べる相手（光を使わない）: 全体の割合だけ（何も見ない）/ 今の行の割合をそのまま写す / 数え方（今の行の品詞の割合と最初の 3 つ → 次の行、の掛け算の表 = 1 層の重み）。
材料: 会話（RealPersonaChat・JMultiWOZ）・青空文庫 300 作品・Web 小説 3000 話（実験だけ、数だけ）。学習用・確かめ用は文書ごとに 9:1。
結果: results/exp84.txt。実行: python experiments/exp84_grammar_synth.py --device cuda --unit line --method 1
"""
import argparse, collections, json, math, os, pickle, random, re, sys, time, zlib

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--unit", default="line")                                                  # line（行）/ punct（句読点の区切り）
ap.add_argument("--method", default="1")                                                   # 1・2・3・4、base（光を使わない比べ）
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--bias", action="store_true")                                             # 直し（2026-11-01）: いつも点いている光（周波数 NL + 1）を入口に足し、全体の割合を覚える道を作る
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import goban_batch
from goban_batch import init_plates, _h, VAL0

sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time(); dev = a.device; PRIV = "D:/AI_git/Sin-CNN3_private"
CACHE = f"{PRIV}/data/views/synth_units.pkl"
L_MAX = 24                                                                                   # 1 単位の語の数の上限（長い行は先頭 24 語）
KEEP = {"助詞", "助動詞"}


def lab(p, w):
    """品詞の値（実験82 と同じ）: 助詞・助動詞は語そのもの、それ以外は大分類-中分類。句読点・括弧は補助記号-…として入る。"""
    if p[0] in KEEP:
        return f"{p[0]}:{w}"
    return p[0] + ("-" + p[1] if p[1] != "*" else "")


# ---- 材料: 行（会話の 1 発話・Web 小説の 1 行・青空文庫の 1 文）ごとに (語, 品詞の値) の並び。一度作ったら保存して使い回す ----
if os.path.exists(CACHE):
    DOCS = pickle.load(open(CACHE, "rb"))
else:
    from sudachipy import Dictionary, SplitMode
    tok = Dictionary(dict="core").tokenizer(); M = SplitMode.C
    DOCS = []                                                                                # [(分け方, [行の (語, 品詞の値) の並び, ...]), ...]

    def add(lines, did):
        rows = [[(m.normalized_form(), lab(m.part_of_speech(), m.normalized_form())) for m in tok.tokenize(x[:300], M)] for x in lines if x.strip()]
        DOCS.append(("test" if zlib.crc32(did.encode()) % 10 == 0 else "train", [r for r in rows if r]))

    for l in open(f"{PRIV}/data/text/conv_all.jsonl", encoding="utf-8"):
        d = json.loads(l)
        if not d["id"].startswith("aozora"):
            add(d["text"].split("\n"), "conv:" + d["id"])
    A = [json.loads(l) for l in open(f"{PRIV}/data/text/aozora_full.jsonl", encoding="utf-8")]; random.Random(0).shuffle(A)
    for d in A[:300]:
        add(re.split(r"(?<=。)", d["text"].replace("\r", "").replace("\n", "").replace("　", "")), "aozora:" + d["id"])
    del A
    W = [json.loads(l) for l in open(f"{PRIV}/data/text/webnovels2.jsonl", encoding="utf-8")]; random.Random(0).shuffle(W)
    for d in W[:3000]:
        add(d["text"].split("\n"), "web:" + d["work"])
    del W
    pickle.dump(DOCS, open(CACHE, "wb"))
print("文書", len(DOCS), f"{time.time() - t0:.0f} 秒", flush=True)
# ---- 単位に分ける（行 / 句読点の区切り）と、品詞の値の一覧 ----
PUNCT = {"、", "。", "！", "？", "!", "?", "，", "．"}


def units(rows):
    if a.unit == "line":
        return rows
    out = []
    for r in rows:
        cur = []
        for w, l in r:
            cur.append((w, l))
            if w in PUNCT:
                out.append(cur); cur = []
        if cur:
            out.append(cur)
    return out


lc = collections.Counter(l for _, rows in DOCS for r in rows for _, l in r if _ == "train" or True)
LABS = [l for l, _ in lc.most_common(80)]; LI = {l: i for i, l in enumerate(LABS)}; NL = len(LABS)   # 品詞の値 80（外は「その他」に当たる番号 NL - 1 にまとめる）
LI_ = lambda l: LI.get(l, NL - 1)
# 重要度: 動詞 = 1、ほかの語 = そのあと 3 語以内に動詞が来る割合（動詞との結びつき）
near = collections.Counter(); seen = collections.Counter()
for s, rows in DOCS:
    if s != "train":
        continue
    for r in rows:
        for k, (w, l) in enumerate(r):
            seen[w] += 1
            if any(r[j][1].startswith("動詞") for j in range(k + 1, min(len(r), k + 4))):
                near[w] += 1
IMP = lambda w, l: 1.0 if l.startswith("動詞-一般") else near[w] / max(1, seen[w])
PAIRS = {"train": [], "test": []}
for s, rows in DOCS:
    U = units(rows)
    for x, y in zip(U, U[1:]):
        PAIRS[s].append((x[:L_MAX], y))
random.Random(0).shuffle(PAIRS["train"]); PAIRS["train"] = PAIRS["train"][:400000]; PAIRS["test"] = PAIRS["test"][:40000]
print("組", len(PAIRS["train"]), len(PAIRS["test"]), "品詞の値", NL, f"{time.time() - t0:.0f} 秒", flush=True)


def tensors(pairs):
    """入力: (組, 位置, 品詞の番号, 重み) / 出力: 次の単位の品詞の割合 (組, NL) と最初の 3 つ (組, 3)（無いときは −1）。"""
    n = len(pairs); X = torch.full((n, L_MAX), -1, dtype=torch.long); Wt = torch.zeros(n, L_MAX)
    Ybag = torch.zeros(n, NL); Yfirst = torch.full((n, 3), -1, dtype=torch.long)
    for i, (x, y) in enumerate(pairs):
        imps = [IMP(w, l) for w, l in x]; mx = max(imps) if imps else 1.0
        for t, (w, l) in enumerate(x):
            X[i, t] = LI_(l); g = imps[t] / mx if mx > 0 else 0.0
            Wt[i, t] = (g if g >= 0.5 else 0.0) if a.method in ("3", "4") else 1.0             # 重要度のグラデーション（足切り 50%）
        for w, l in y:
            Ybag[i, LI_(l)] += 1
        for t, (w, l) in enumerate(y[:3]):
            Yfirst[i, t] = LI_(l)
    return X, Wt, Ybag / Ybag.sum(1, keepdim=True).clamp(min=1), Yfirst


Xtr, Wtr, Btr, Ftr = tensors(PAIRS["train"]); Xte, Wte, Bte, Fte = tensors(PAIRS["test"])
print("テンソル", f"{time.time() - t0:.0f} 秒", flush=True)
res = dict(exp="84", bias=a.bias, unit=a.unit, method=a.method, pairs=len(PAIRS["train"]), labels=NL, steps=a.steps)
bits_bag = lambda P, Y: float(-(Y * torch.log2(P.clamp(min=1e-9))).sum() / Y.shape[0])
acc_first = lambda P3, F: [float((P3[:, t].argmax(-1) == F[:, t])[F[:, t] >= 0].float().mean()) for t in range(3)]
if a.method == "base":
    prior = Btr.mean(0); res["prior_bits"] = bits_bag(prior.expand_as(Bte), Bte)
    cur = torch.zeros_like(Bte)
    for i in range(len(Xte)):
        for t in range(L_MAX):
            if Xte[i, t] >= 0:
                cur[i, Xte[i, t]] += 1
    cur = cur / cur.sum(1, keepdim=True).clamp(min=1); res["copy_bits"] = bits_bag(0.7 * cur + 0.3 * prior, Bte)
    fprior = torch.stack([torch.bincount(Ftr[:, t][Ftr[:, t] >= 0], minlength=NL).float() for t in range(3)])
    res["first_prior_acc"] = [float((Fte[:, t] == fprior[t].argmax())[Fte[:, t] >= 0].float().mean()) for t in range(3)]
    # 数え方（1 層の重み）: 入力 = 今の単位の品詞の割合 + 最初の 3 つ（one-hot）→ 出力 = 次の単位の割合・最初の 3 つ（softmax）
    def feats(X):
        F = torch.zeros(len(X), NL * 4)
        for t in range(L_MAX):
            m = X[:, t] >= 0; F[m, X[m, t]] += 1.0 / L_MAX
            if t < 3:
                F[m, NL * (t + 1) + X[m, t]] = 1.0
        return F
    Ftr_x, Fte_x = feats(Xtr), feats(Xte)
    Wl = torch.zeros(NL * 4, NL * 4, requires_grad=True, device=dev); bl = torch.zeros(NL * 4, requires_grad=True, device=dev)
    opt = torch.optim.Adam([Wl, bl], lr=0.01)
    for st in range(a.steps):
        idx = torch.randint(0, len(Ftr_x), (a.batch,)); o = Ftr_x[idx].to(dev) @ Wl + bl
        loss = -(Btr[idx].to(dev) * torch.log_softmax(o[:, :NL], -1)).sum(-1).mean()
        for t in range(3):
            m = (Ftr[idx, t] >= 0).to(dev)
            loss = loss + torch.nn.functional.cross_entropy(o[:, NL * (t + 1):NL * (t + 2)][m], Ftr[idx, t].to(dev)[m])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        o = (Fte_x.to(dev) @ Wl + bl).cpu()
    res["linear_bits"] = bits_bag(torch.softmax(o[:, :NL], -1), Bte)
    res["linear_first_acc"] = acc_first(torch.stack([torch.softmax(o[:, NL * (t + 1):NL * (t + 2)], -1) for t in range(3)], 1), Fte)
else:
    N, K = 512, 16; H = _h(dev, N)
    y = torch.arange(N, dtype=torch.float32, device=dev)
    WAVE = torch.stack([torch.exp(1j * 2 * math.pi * (k + 1) * y / N) / math.sqrt(N) for k in range(NL + 1)]).to(torch.complex64)   # 品詞 k = 周波数 k + 1（最後 = いつも点いている光）
    conv = a.method in ("2", "4")
    PH = torch.exp(1j * 2 * math.pi * torch.arange(L_MAX + 2, device=dev) / 25).to(torch.complex64)   # 位置の位相

    def encode(X, Wt):
        """並びを入口の光にする: Σ_t 重み × 品詞の波 × 位置の位相（畳み込みなら となりの位置にも広げる）。"""
        X = X.to(dev); Wt = Wt.to(dev); m = (X >= 0).float() * Wt
        ph = PH[1:L_MAX + 1][None]
        if conv:
            ph = 0.5 * PH[0:L_MAX][None] + ph + 0.5 * PH[2:L_MAX + 2][None]
        coef = (m * ph)                                                                      # (組, 位置) の複素の重み
        u = torch.einsum("bt,btn->bn", coef, WAVE[X.clamp(min=0)])
        u = u / u.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-9)                 # 光の強さをそろえる
        if a.bias:
            u = (u + WAVE[NL][None]) / math.sqrt(2)                                          # いつも点いている光（全体の割合の道）
        return u

    def forward(P, u):
        u = torch.nn.functional.pad(u, (0, N))
        for k in range(K):
            u = torch.cat([u[..., :N] * torch.exp(1j * P[:, k]), u[..., N:]], -1)
            u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
        I = torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs() ** 2
        out = I[:, VAL0:VAL0 + 4 * NL].reshape(-1, 4, NL)                                  # 割合・1 番目・2 番目・3 番目（グループごとに割合に直す）
        return out / out.sum(-1, keepdim=True).clamp(min=1e-12)

    P = init_plates(1, 0, dev, N, K)[0].requires_grad_(True)
    opt = torch.optim.Adam([P], lr=0.02); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)   # 学習率を途中で下げる（直し）
    for st in range(a.steps):
        idx = torch.randint(0, len(Xtr), (a.batch,))
        o = forward(P, encode(Xtr[idx], Wtr[idx]))
        loss = -(Btr[idx].to(dev) * torch.log(o[:, 0] + 1e-9)).sum(-1).mean()
        for t in range(3):
            m = (Ftr[idx, t] >= 0).to(dev)
            loss = loss - torch.log(o[:, t + 1].gather(-1, Ftr[idx, t].clamp(min=0).to(dev)[:, None])[:, 0] + 1e-9)[m].mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if st % 500 == 0:
            print(a.unit, a.method, st, "回", round(float(loss), 4), f"{time.time() - t0:.0f} 秒", flush=True)
    with torch.no_grad():
        O = torch.cat([forward(P, encode(Xte[i:i + 2000], Wte[i:i + 2000])).cpu() for i in range(0, len(Xte), 2000)])
    O = 0.98 * O + 0.02 / NL
    res["bits"] = bits_bag(O[:, 0], Bte); res["first_acc"] = acc_first(O[:, 1:], Fte)
    torch.save(dict(P=P.detach().cpu(), labels=LABS, unit=a.unit, method=a.method), f"{PRIV}/models/synth_{a.unit}_{a.method}.pt")
res["sec"] = time.time() - t0
with open("results/exp84.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print(res)
