"""実験21: 観点周波数（持ち主の「属性選択周波数」）— 物の属性を順引きし、属性から物を逆引きする（リンゴとトマト）

着想: 2026-10-27 持ち主の案。縦の周波数のずらし（観点）ごとに意味（色・大きさ・形・味・触感）を与え、物を入れると、その観点の値の光が光る（順引き）。
      逆に、値の光から物へたどり（逆引き）、観点ごとの結果を掛け算すると、どの物の可能性が高いかが出るのでは。
      例: 「丸い赤い実をむしゃむしゃ食べた、酸っぱかった」→ トマトのようなもの。
方法: 記号 17 個（物 5: リンゴ トマト バナナ レモン イチゴ、値 12: 赤 緑 黄 大 小 丸い 細長い 甘い 酸っぱい しゃきしゃき むしゃむしゃ ねっとり）に、
      周波数 −48, −42, …, 48（6 の倍数）の波を割り当てる。観点 = 縦の周波数のずらし: 色 1・大きさ 2・形 3・味 4・触感 5（ずらした先は記号と重ならない）。
      順引きの学習: 入力 = 物の波を観点だけずらしたもの、答え = その観点の値の分布（例: リンゴの色は 赤 80%・緑 20%。持ち主の例にならい私が決めた表 TABLE）。
      器は実験16 と同じ（列の間 64 画素、周波数 1〜16、縦横とも sin と cos、曲がる処理なし）。Adam（0.01）1500 回、3 seed。
      逆引き: 値の波を出口から逆向きに通す（光の器の逆の計算 = 各段の逆を逆の順に）。入口で「物の周波数 + 観点のずらし」の明るさを読む。
      式の予想: 曲がる処理のない光の器では、逆向きの明るさ = 順引きでその物がその値に送る明るさ（光の量を保つ器の性質）。
      問い合わせ: 観点ごとの逆引きの明るさを掛け算して、物ごとの点数にする（どの物の可能性が高いか）。比較相手: 表 TABLE から計算した正解の点数。
指標: 順引きの分布と表の差（全変動距離 = 分布の差の絶対値の和 ÷ 2）、逆引きと順引きの差、問い合わせで一番の物が正解と同じか。
結果の置き場: results/exp21.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp21_viewpoint.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--F", type=int, default=16)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp21.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, F, L = a.N, a.F, 2 * a.N
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
OBJ = ["リンゴ", "トマト", "バナナ", "レモン", "イチゴ"]
VAL = ["赤", "緑", "黄", "大", "小", "丸い", "細長い", "甘い", "酸っぱい", "しゃきしゃき", "むしゃむしゃ", "ねっとり"]
SYM = OBJ + VAL; freq = {s: -48 + 6 * i for i, s in enumerate(SYM)}
VIEW = {"色": 1, "大きさ": 2, "形": 3, "味": 4, "触感": 5}                               # 観点 = 縦の周波数のずらし
TABLE = {"リンゴ": {"色": {"赤": .8, "緑": .2}, "大きさ": {"大": .6, "小": .4}, "形": {"丸い": 1}, "味": {"甘い": .6, "酸っぱい": .4}, "触感": {"しゃきしゃき": 1}},
         "トマト": {"色": {"赤": .9, "緑": .1}, "大きさ": {"大": .4, "小": .6}, "形": {"丸い": 1}, "味": {"甘い": .3, "酸っぱい": .7}, "触感": {"むしゃむしゃ": 1}},
         "バナナ": {"色": {"黄": 1}, "大きさ": {"大": .7, "小": .3}, "形": {"細長い": 1}, "味": {"甘い": 1}, "触感": {"ねっとり": 1}},
         "レモン": {"色": {"黄": .9, "緑": .1}, "大きさ": {"大": .2, "小": .8}, "形": {"丸い": .7, "細長い": .3}, "味": {"酸っぱい": 1}, "触感": {"しゃきしゃき": .5, "むしゃむしゃ": .5}},
         "イチゴ": {"色": {"赤": 1}, "大きさ": {"小": 1}, "形": {"丸い": .8, "細長い": .2}, "味": {"甘い": .7, "酸っぱい": .3}, "触感": {"むしゃむしゃ": 1}}}
QUERIES = {"丸い赤い実をむしゃむしゃ食べた、酸っぱかった（持ち主の例）": {"形": "丸い", "色": "赤", "触感": "むしゃむしゃ", "味": "酸っぱい"},
           "赤くて丸い": {"色": "赤", "形": "丸い"},
           "黄色くて細長い": {"色": "黄", "形": "細長い"},
           "赤くてしゃきしゃき、甘い": {"色": "赤", "触感": "しゃきしゃき", "味": "甘い"}}
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)


def grid(A):
    return sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))


def forward(A, u0):
    G = grid(A); u = torch.nn.functional.pad(u0, (0, L - N))
    for col in range(N):
        u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return u


def backward(A, uL):
    """出口の場 (n, L) を逆向きに通す: 各段の逆（回折の逆 → 板の逆）を逆の順に。入口の場 (n, N) を返す。"""
    G = grid(A); u = uL
    for col in reversed(range(N)):
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * torch.conj(H), n=L, dim=-1, norm="ortho")
        u = torch.cat([u[:, :N] * torch.exp(-1j * G[:, col]), u[:, N:]], 1)
    return u[:, :N]


spec = lambda u: torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs() ** 2
pairs = [(o, v) for o in OBJ for v in VIEW]
u0 = torch.stack([wave(freq[o] + VIEW[v]) for o, v in pairs])
q = torch.zeros(len(pairs), len(SYM) + 1)
for n, (o, v) in enumerate(pairs):
    for val, p in TABLE[o][v].items():
        q[n, SYM.index(val)] = p
bins = torch.tensor([freq[s] % N for s in SYM])
res = dict(exp="21", args=vars(a), rows=[])
for seed in range(a.seeds):
    t0 = time.time(); g = torch.Generator().manual_seed(seed)
    A = torch.nn.Parameter(0.1 * torch.randn(4, F, F, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
    for step in range(a.steps):
        W = spec(forward(A, u0))[:, bins]
        p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        W = spec(forward(A, u0))[:, bins]; D = W / W.sum(1, keepdim=True)                   # 順引き: 値の分布
        tv = float((D - q[:, :-1]).abs().sum(1).mean() / 2); light = float(W.sum(1).mean())
        back, fwd = {}, {}                                                                 # 逆引き（値 × 観点 → 物ごとの明るさ）と、順引きから見た同じ量
        for v, d in VIEW.items():
            for val in VAL:
                uL = torch.nn.functional.pad(wave(freq[val])[None], (0, L - N))
                S = spec(backward(A, uL))[0]
                back[(v, val)] = [float(S[(freq[o] + d) % N]) for o in OBJ]
                fwd[(v, val)] = [float(W[pairs.index((o, v)), SYM.index(val)]) for o in OBJ]
        err_back = max(abs(x - y) for k in back for x, y in zip(back[k], fwd[k]))
        qres = {}
        for name, ev in QUERIES.items():
            score = np.prod([back[(v, val)] for v, val in ev.items()], axis=0)
            truth = np.prod([[TABLE[o][v].get(val, 0) for o in OBJ] for v, val in ev.items()], axis=0)
            qres[name] = dict(score=(score / score.sum()).tolist(), truth=(truth / truth.sum()).tolist(),
                              top=OBJ[int(np.argmax(score))], top_truth=OBJ[int(np.argmax(truth))])
    r = dict(seed=seed, tv=tv, light=light, err_back=err_back, queries=qres, dist=D.tolist(), sec=time.time() - t0); res["rows"].append(r)
    print(f"seed {seed}: 順引きの分布と表の差 {tv:.3f}  答えの値に入った光 {100 * light:.1f}%  逆引き − 順引き {err_back:.1e}  {r['sec']:.0f} 秒")
    for name, x in qres.items():
        print(f"   「{name}」→ {x['top']}（正解 {x['top_truth']}）  点数 " + " ".join(f"{o}{100 * s:.0f}%" for o, s in zip(OBJ, x["score"])))
res.update(OBJ=OBJ, VAL=VAL, numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
