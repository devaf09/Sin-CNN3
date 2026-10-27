"""実験27: 関係性の関係 — 知らない意味に気づき、何の仲間かを当て、新しい意味を書き込む（リンゴさん）

着想: 2026-10-27 持ち主の案「リンゴさんを知らなくても、既存の果物リンゴに合わないとわかれば、それが別の属性で作られたリンゴという何か、とわかる。
      関係性の関係という考え方を追加すると、それがすぐに人間のリンゴさんって分かるんじゃないかな。
      例: リンゴは楽しそうに手を振った。→ 手を振る: 人間、楽しそう: 人間の確率高い → りんごは人間の属性を持つ」。
方法: 1 つの光の器（実験21・26 と同じ作り、縦 128 画素、記号 22 個 = 周波数 −63, −57, …, 63、曲がる処理なし）に 3 段の関係を覚えさせる。
        ずらし 0（意味）: 文字列 → 物。「リンゴ」→ 果物のリンゴ（人の意味は知らない）、「ミカン」「太郎」「ポチ」も 1 つずつ。
        ずらし 1（仲間）: 物 → 仲間（果物・人間・動物）。「関係性の関係」の上の段。
        ずらし 2〜4（行い・様子・味）: 仲間 → 値（属性は物ではなく仲間に覚えさせる。私が決めた表）。
      (1) 驚き: 文を「文字列 → 物 → 仲間 → 手がかり」とたどった明るさ（どれだけ合うか）を、文ごとに比べる。とても暗ければ「知っている意味に合わない」。
      (2) 仲間当て: 手がかりだけを逆向きに通して掛け算し、仲間の点数を出す。
      (3) 書き込み: 空いている記号「未知 1」を使い、「リンゴ」→ 果物のリンゴ 50%・未知 1 50%、未知 1 → (2) で当てた仲間、を足して少しだけ学習し直す（200 回）。
          前の関係が保たれるか（表との差）、書き込み後に「リンゴが話した」の意味が未知 1 になるかを見る。
結果の置き場: results/exp27.txt。3 seed。
実行: CUDA_VISIBLE_DEVICES= python experiments/exp27_unknown_sense.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=128)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--steps2", type=int, default=200)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp27.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
WORD = ["「リンゴ」", "「ミカン」", "「太郎」", "「ポチ」"]
OBJ = ["果物のリンゴ", "果物のミカン", "人の太郎", "犬のポチ", "未知 1"]
CAT = ["果物", "人間", "動物"]
VAL = ["食べた", "切った", "手を振った", "話した", "吠えた", "走った", "楽しそう", "静か", "甘い", "酸っぱい"]
SYM = WORD + OBJ + CAT + VAL; freq = {s: -63 + 6 * i for i, s in enumerate(SYM)}
VIEW = {"意味": 0, "仲間": 1, "行い": 2, "様子": 3, "味": 4}
VALS_OF = {"行い": VAL[:6], "様子": ["楽しそう", "静か"], "味": ["甘い", "酸っぱい"]}
diffuse = lambda v: {x: 1 / len(VALS_OF[v]) for x in VALS_OF[v]}
REL = {("「リンゴ」", "意味"): {"果物のリンゴ": 1}, ("「ミカン」", "意味"): {"果物のミカン": 1}, ("「太郎」", "意味"): {"人の太郎": 1}, ("「ポチ」", "意味"): {"犬のポチ": 1},
       ("果物のリンゴ", "仲間"): {"果物": 1}, ("果物のミカン", "仲間"): {"果物": 1}, ("人の太郎", "仲間"): {"人間": 1}, ("犬のポチ", "仲間"): {"動物": 1},
       ("果物", "行い"): {"食べた": .6, "切った": .4}, ("人間", "行い"): {"手を振った": .3, "話した": .4, "走った": .3}, ("動物", "行い"): {"吠えた": .5, "走った": .5},
       ("果物", "様子"): {"静か": 1}, ("人間", "様子"): {"楽しそう": .6, "静か": .4}, ("動物", "様子"): {"楽しそう": .5, "静か": .5},
       ("果物", "味"): {"甘い": .6, "酸っぱい": .4}, ("人間", "味"): diffuse("味"), ("動物", "味"): diffuse("味")}
SENT = {"リンゴを食べた": ("「リンゴ」", {"行い": "食べた"}), "リンゴは甘い": ("「リンゴ」", {"味": "甘い"}),
        "リンゴは楽しそうに手を振った（持ち主の例）": ("「リンゴ」", {"行い": "手を振った", "様子": "楽しそう"}),
        "太郎は楽しそうに手を振った": ("「太郎」", {"行い": "手を振った", "様子": "楽しそう"}), "ポチが吠えた": ("「ポチ」", {"行い": "吠えた"})}
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)
bins = torch.tensor([freq[s] % N for s in SYM])


def grid(A):
    return sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))


def prop(A, u, back=False):
    G = grid(A); u = torch.nn.functional.pad(u, (0, L - N)) if u.shape[-1] == N else u
    for col in (reversed(range(N)) if back else range(N)):
        if back:
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * torch.conj(H), n=L, dim=-1, norm="ortho")
            u = torch.cat([u[:, :N] * torch.exp(-1j * G[:, col]), u[:, N:]], 1)
        else:
            u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs() ** 2               # 周波数ごとの明るさ


def train(A, rel, steps):
    keys = list(rel); u0 = torch.stack([wave(freq[s] + VIEW[v]) for s, v in keys])
    q = torch.zeros(len(keys), len(SYM) + 1)
    for n, k in enumerate(keys):
        for x, p in rel[k].items():
            q[n, SYM.index(x)] = p
    opt = torch.optim.Adam([A], lr=0.01)
    for step in range(steps):
        W = prop(A, u0)[:, bins]; p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        W = prop(A, u0)[:, bins]; D = W / W.sum(1, keepdim=True)
    return float((D - q[:, :-1]).abs().sum(1).mean() / 2)


def read(A, src, view, targets, back=False):
    """順向き: src をずらして入れ、targets の明るさ。逆向き: 値 src の光を逆に通し、targets（+ ずらし）の明るさ。"""
    with torch.no_grad():
        if back:
            S = prop(A, wave(freq[src])[None], back=True)[0]; return np.array([float(S[(freq[t] + VIEW[view]) % N]) for t in targets])
        S = prop(A, wave(freq[src] + VIEW[view])[None])[0]; return np.array([float(S[freq[t] % N]) for t in targets])


def analyse(A, name, w, ev):
    objs = read(A, w, "意味", OBJ)                                                       # 文字列 → 物
    cat_of = np.array([read(A, o, "仲間", CAT) for o in OBJ])                           # 物 → 仲間
    like = np.prod([read(A, val, v, CAT, back=True) for v, val in ev.items()], axis=0)  # 手がかり → 仲間（逆向きの掛け算）
    fit = float(objs @ cat_of @ like)                                                    # 文字列 → 物 → 仲間 → 手がかり とたどった明るさ
    sense = objs * (cat_of @ like); sense = sense / max(sense.sum(), 1e-30)
    return dict(fit=fit, cat=(like / like.sum()).tolist(), sense=sense.tolist())


res = dict(exp="27", args=vars(a), rows=[])
for seed in range(a.seeds):
    t0 = time.time(); g = torch.Generator().manual_seed(seed)
    A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); tv0 = train(A, REL, a.steps)
    before = {nm: analyse(A, nm, w, ev) for nm, (w, ev) in SENT.items()}
    newcat = CAT[int(np.argmax(before["リンゴは楽しそうに手を振った（持ち主の例）"]["cat"]))]
    REL2 = dict(REL); REL2[("「リンゴ」", "意味")] = {"果物のリンゴ": .5, "未知 1": .5}; REL2[("未知 1", "仲間")] = {newcat: 1}
    tv_old = train(A, REL2, a.steps2)                                                   # 少しだけ学習し直す
    after = {nm: analyse(A, nm, w, ev) for nm, (w, ev) in list(SENT.items()) + [("リンゴが話した", ("「リンゴ」", {"行い": "話した"}))]}
    r = dict(seed=seed, tv=tv0, tv_after=tv_old, new_category=newcat, before=before, after=after, sec=time.time() - t0); res["rows"].append(r)
    print(f"seed {seed}: 表との差 {tv0:.3f}（書き込み後 {tv_old:.3f}）  当てた仲間: {newcat}  {r['sec']:.0f} 秒")
    for nm, x in before.items():
        print(f"   前「{nm}」 たどった明るさ {x['fit']:.2e}  仲間 " + " ".join(f"{c}{100 * p:.0f}" for c, p in zip(CAT, x["cat"])))
    for nm in ("リンゴを食べた", "リンゴは楽しそうに手を振った（持ち主の例）", "リンゴが話した"):
        x = after[nm]; print(f"   後「{nm}」 たどった明るさ {x['fit']:.2e}  意味 " + " ".join(f"{o}{100 * p:.0f}" for o, p in zip(OBJ, x["sense"]) if p > 0.01))
res.update(OBJ=OBJ, CAT=CAT, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
