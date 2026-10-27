"""実験32: 道具の要素 × 単語の格子 → 結果（持ち主の案）— 見せていない道具でも、要素から推し量れるか

着想: 2026-10-27 持ち主の案「入側に『ミキサー』という要素・属性があって、それがリンゴの格子に入るとジュースの属性になる。『プレス機』でもジュースになる。
      包丁・ピーラーに関連する要素をもつ何かがあれば、リンゴの格子からは皮をむく・調理・食べる、といった要素が出る」。
      実験31 の「単語ごとに格子 1 枚」を使い、入口に「何か」の代わりに道具の要素の光を入れる。単語の格子が「関数」になる形。
方法: 道具の要素 5 つ（刃・回転・圧力・熱・薄く）、結果 7 つ（ジュース・細かい・皮をむく・切る・煮る・焼く・食べる）を周波数の波にする（縦 100）。
      道具 = 要素の光の重ね合わせ: ミキサー = 刃 + 回転、プレス機 = 圧力、包丁 = 刃、ピーラー = 刃 + 薄く、鍋 = 熱。
      単語 3 つ（リンゴ・ニンジン・肉）の格子（1 単語 1 KB）を、私が決めた表 TABLE で学習（実験31 と同じ器、曲がる処理なし、1500 回、3 seed）。
      見せていない道具: フードプロセッサ = 刃 + 回転（ミキサーと同じ要素で名前だけ違う）、圧力鍋 = 圧力 + 熱（見せていない組み合わせ）、
      ホットプレート = 熱（鍋と同じ要素）。
      逆引き: 結果「ジュース」の光をリンゴの格子に逆向きに通し、入口で要素ごとの明るさを読む（何を使えばジュースになるか）。
結果の置き場: results/exp32.txt
実行: CUDA_VISIBLE_DEVICES= nice -n 10 python experiments/exp32_tools.py
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--N", type=int, default=100)
ap.add_argument("--Fw", type=int, default=8)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp32.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
N, L = a.N, 2 * a.N
ELEM = ["刃", "回転", "圧力", "熱", "薄く"]; RES = ["ジュース", "細かい", "皮をむく", "切る", "煮る", "焼く", "食べる"]
freq = {e: -40 + 8 * i for i, e in enumerate(ELEM)}; freq.update({r: 6 + 6 * i for i, r in enumerate(RES)})
TOOL = {"ミキサー": ["刃", "回転"], "プレス機": ["圧力"], "包丁": ["刃"], "ピーラー": ["刃", "薄く"], "鍋": ["熱"]}
NEW = {"フードプロセッサ": ["刃", "回転"], "圧力鍋": ["圧力", "熱"], "ホットプレート": ["熱"]}
TABLE = {"リンゴ": {"ミキサー": {"ジュース": .7, "細かい": .3}, "プレス機": {"ジュース": 1}, "包丁": {"切る": .6, "食べる": .4}, "ピーラー": {"皮をむく": .8, "食べる": .2}, "鍋": {"煮る": .7, "食べる": .3}},
         "ニンジン": {"ミキサー": {"ジュース": .5, "細かい": .5}, "プレス機": {"ジュース": 1}, "包丁": {"切る": .7, "煮る": .3}, "ピーラー": {"皮をむく": 1}, "鍋": {"煮る": 1}},
         "肉": {"ミキサー": {"細かい": 1}, "プレス機": {"細かい": .6, "焼く": .4}, "包丁": {"切る": 1}, "ピーラー": {"切る": .5, "細かい": .5}, "鍋": {"煮る": .6, "焼く": .4}}}
pos = torch.arange(N, dtype=torch.float32); fr = torch.arange(1, a.Fw + 1, dtype=torch.float32)
sy, cy = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
wave = lambda f: (torch.exp(1j * 2 * math.pi * f * pos / N) / math.sqrt(N)).to(torch.complex64)
tool_light = lambda els: sum(wave(freq[e]) for e in els) / math.sqrt(len(els))        # 道具 = 要素の光の重ね合わせ（明るさの合計 1）
rb = torch.tensor([freq[r] % N for r in RES])


def prop(G, u, back=False):
    u = torch.nn.functional.pad(u, (0, L - N))
    for col in (reversed(range(N)) if back else range(N)):
        if back:
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * torch.conj(H), n=L, dim=-1, norm="ortho")
            u = torch.cat([u[:, :N] * torch.exp(-1j * G[:, col]), u[:, N:]], 1)
        else:
            u = torch.cat([u[:, :N] * torch.exp(1j * G[:, col]), u[:, N:]], 1)
            u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs() ** 2


res = dict(exp="32", args=vars(a), RES=RES, ELEM=ELEM, rows=[])
for seed in range(a.seeds):
    t0 = time.time(); g = torch.Generator().manual_seed(seed); out = {}
    for w, tab in TABLE.items():
        tools = list(tab); u0 = torch.stack([tool_light(TOOL[t]) for t in tools])
        q = torch.zeros(len(tools), len(RES) + 1)
        for i, t in enumerate(tools):
            for r, p in tab[t].items():
                q[i, RES.index(r)] = p
        A = torch.nn.Parameter(0.1 * torch.randn(4, a.Fw, a.Fw, generator=g)); opt = torch.optim.Adam([A], lr=0.01)
        grid = lambda: sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((sy, cy), (sy, sy), (cy, cy), (cy, sy))))
        for step in range(a.steps):
            W = prop(grid(), u0)[:, rb]; p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
            loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            G = grid(); W = prop(G, u0)[:, rb]; D = W / W.sum(1, keepdim=True)
            tv = float((D - q[:, :-1]).abs().sum(1).mean() / 2)
            newW = prop(G, torch.stack([tool_light(v) for v in NEW.values()]))[:, rb]
            new = {t: (x / x.sum()).tolist() for t, x in zip(NEW, newW)}
            S = prop(G, wave(freq["ジュース"])[None], back=True)[0]                          # 逆引き: ジュースの光を逆向きに
            juice = {e: float(S[freq[e] % N]) for e in ELEM}; tot = sum(juice.values()); juice = {e: v / tot for e, v in juice.items()}
        out[w] = dict(tv=tv, new=new, juice_from=juice)
    r = dict(seed=seed, words=out, sec=time.time() - t0); res["rows"].append(r)
    print(f"seed {seed}（{r['sec']:.0f} 秒）")
    for w, x in out.items():
        top = lambda d: " ".join(f"{RES[i]}{100 * d[i]:.0f}" for i in np.argsort(d)[::-1][:3])
        print(f"  {w}: 表との差 {x['tv']:.3f}  " + "  ".join(f"{t}→{top(np.array(d))}" for t, d in x["new"].items()) +
              "  ジュースの逆引き: " + " ".join(f"{e}{100 * v:.0f}" for e, v in sorted(x["juice_from"].items(), key=lambda z: -z[1])))
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
