"""実験28（大規模実験 B-1・B-2）: 規模を広げる — 青空文庫の名詞 80 × 観点 9 を 1 つの光の器に覚えさせ、簡単な質問に答える

着想: 2026-10-27 持ち主の依頼「ちょっと大規模な調査と実験をしてみようか？まだチャットボットにはならないかもだけど、何が足りないかわかるから」。
      実験21・25 は物 5〜12 個だった。記号を増やすと、器（学習する数）と光の場（長さ）がどこまで足りるかを測る。
方法: tools/aozora_adj_nouns.py で数えた「形容詞 + 名詞」から、具体的な名詞 80 個（数の多い順、言葉の切れ端を除く）と値 23 個を使う。
      観点 9 つ（色・大きさ・温度・味・長さ・高さ・重さ・明るさ・新しさ）= 縦の周波数のずらし 1〜9。記号 103 個 = 周波数 −510, −500, …（10 おき）。
      縦の長さ N = 1040。数えた数が 3 未満の観点は拡散（分からない）。器は実験27 と同じ作り（曲がる処理なし）だが、光が通る列は --cols（既定 20、時間のため）。
      格子の周波数 F（学習する数 = 4F²）を 16・32・64 で比べる。1500 回、seed 0（時間のため 1 本。3 本は後で）。
指標: 順引きの分布と数えた割合の差（全変動距離）、
      質問 1「雪は何色？」型: (名詞, 観点) ごとに一番明るい値 = 数えた割合で一番の値 か（データのある組だけ）、
      質問 2「白いものは？」型: 値ごとに逆引きで上位 5 個の名詞と、数えた割合（その値の割合）の上位 5 個の重なり、学習の時間。
結果の置き場: results/exp28.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp28_scale_nouns.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=1040)
ap.add_argument("--Fs", default="16,32,64", help="格子の周波数の数。0 なら画素ごとに自由な位相の板（回折の深層ネットワークと同じ、学習する数 = N × 列）")
ap.add_argument("--cols", type=int, default=20)
ap.add_argument("--init", default="0.1", help="格子の強さの初期値の大きさ。'scaled' なら 0.1/F（格子の値の大きさが F によらずそろう）")
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--n_nouns", type=int, default=80)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp28.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2); dev = a.device
C = json.load(open("results/aozora_adj_nouns.json", encoding="utf-8"))
SKIP = set("間 方 物 所 中 大 時 長 処 事 小 日 黒 白 面 笑 子 美 奴 調子 気持 話 力 心 言葉 色 音 形 声 様子 上 下 前 後 外 内 他 方々 程 位 者 人 一 二 三".split())
NOUN = [n for n in C["nouns"] if n not in SKIP][:a.n_nouns]; VALS_OF = C["VIEW"]; VAL = [x for v in VALS_OF.values() for x in v]
VIEW = {v: i + 1 for i, v in enumerate(VALS_OF)}; SYM = NOUN + VAL
N, L = a.N, 2 * a.N; freq = {s: 10 * (i - len(SYM) // 2) for i, s in enumerate(SYM)}
TABLE = {n: {v: ({k: c / sum(C["count"][n][v].values()) for k, c in C["count"][n][v].items()} if sum(C["count"][n][v].values()) >= 3 else None) for v in VIEW} for n in NOUN}
pos = torch.arange(N, dtype=torch.float32, device=dev)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L, device=dev)) ** 2))
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)
bins = torch.tensor([freq[s] % N for s in SYM], device=dev)


def prop(G, u, back=False):
    u = torch.nn.functional.pad(u, (0, L - N))
    for col in (reversed(range(a.cols)) if back else range(a.cols)):
        if back:
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * torch.conj(H), n=L, dim=-1, norm="ortho")
            u = torch.cat([u[:, :N] * torch.exp(-1j * G[:, col]), u[:, N:]], 1)
        else:
            u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs() ** 2


pairs = [(n, v) for n in NOUN for v in VIEW]
u0 = torch.stack([wave(freq[n] + VIEW[v]) for n, v in pairs])
q = torch.zeros(len(pairs), len(SYM) + 1, device=dev)
for k, (n, v) in enumerate(pairs):
    t = TABLE[n][v] or {x: 1 / len(VALS_OF[v]) for x in VALS_OF[v]}
    for x, p in t.items():
        q[k, SYM.index(x)] = p
res = dict(exp="28", args=vars(a), source=C["source"], nouns=NOUN, rows=[])
for F in [int(f) for f in a.Fs.split(",")]:
    t0 = time.time(); g = torch.Generator().manual_seed(a.seed)
    fr = torch.arange(1, F + 1, dtype=torch.float32, device=dev)
    sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * torch.arange(a.cols, device=dev) / a.cols)
    sx = torch.sin(2 * math.pi * fr[:, None] * torch.arange(a.cols, device=dev) / a.cols); cyy = torch.cos(2 * math.pi * fr[:, None] * pos / N)
    if F == 0:                                                                          # 画素ごとに自由な位相（周波数の制限なし）
        A = torch.nn.Parameter((0.1 * torch.randn(N, a.cols, generator=g)).to(dev)); grid = lambda: A
    else:
        s0 = 0.1 / F if a.init == "scaled" else float(a.init)                                 # F が大きいと、4F² 個を足した格子の値が大きくなりすぎる
        A = torch.nn.Parameter((s0 * torch.randn(4, F, F, generator=g)).to(dev))
        grid = lambda: sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sx), (cyy, cy), (cyy, sx))))   # 縦 (N) × 列 (cols)
    opt = torch.optim.Adam([A], lr=0.01)
    for step in range(a.steps):
        W = prop(grid(), u0)[:, bins]; p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
        loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        G = grid(); W = prop(G, u0)[:, bins]; D = (W / W.sum(1, keepdim=True)).cpu().numpy(); Q = q[:, :-1].cpu().numpy()
        tv = float(np.abs(D - Q).sum(1).mean() / 2)
        known = [k for k, (n, v) in enumerate(pairs) if TABLE[n][v]]
        vi = {v: [SYM.index(x) for x in VALS_OF[v]] for v in VIEW}                          # その観点の値の中だけで比べる
        q1 = float(np.mean([np.argmax(D[k][vi[pairs[k][1]]]) == np.argmax(Q[k][vi[pairs[k][1]]]) for k in known]))   # 「雪は何色？」型
        base = float(np.mean([np.argmax(Q[k][vi[pairs[k][1]]]) == np.argmax(sum(Q[j][vi[pairs[k][1]]] for j in known if pairs[j][1] == pairs[k][1])) for k in known]))   # 比較: 観点ごとに一番多い値
        q2 = []
        for v, d in VIEW.items():                                                              # 「白いものは？」型
            for val in VALS_OF[v]:
                S = prop(G, wave(freq[val])[None], back=True)[0]
                sc = np.array([float(S[(freq[n] + d) % N]) for n in NOUN])
                tr = np.array([(TABLE[n][v] or {}).get(val, 0.0) for n in NOUN])
                if (tr > 0).sum() >= 5:
                    q2.append(len(set(np.argsort(sc)[::-1][:5]) & set(np.argsort(tr)[::-1][:5])) / 5)
        light = float(W.sum(1).mean())
    r = dict(F=F, params=A.numel(), tv=tv, q1=q1, q1_majority=base, q2=float(np.mean(q2)), n_q2=len(q2), n_known=len(known), light=light, sec=time.time() - t0)
    res["rows"].append(r)
    print(f"F={F if F else '画素':<3} 学習する数 {A.numel():>6}  表との差 {tv:.3f}  質問 1（何色？）{100 * q1:.1f}%（観点ごとに一番多い値 {100 * base:.1f}%、{len(known)} 組）  質問 2（白いものは？上位 5 の重なり）{100 * r['q2']:.1f}%（{len(q2)} 問）  "
          f"答えの記号に入った光 {100 * light:.1f}%  {r['sec']:.0f} 秒", flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
