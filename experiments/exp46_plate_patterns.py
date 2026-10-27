"""実験46: 共通の板（初期の板）の型を増やして、学習の経過を全部記録する（易しい課題・重い課題、GPU）

着想: 2026-10-28 持ち主「板のパターンは多くなるかもだけど、データ取り（学習回数と結果とか）もおねがいね、GPU もどんどん使って」。実験45 の続き。
方法: 画素ごとの板（列 20、列の間 64 画素）。初期の板の型 15 種（下の PLATES）。課題 2 つ:
        易しい = 実験21 の表（物 5・観点 5・値 12、使う周波数 37 個を低い所に詰める、縦 128）、600 回
        重い   = 実験28 の名詞 80 × 観点 9（青空文庫の形容詞、記号 103、縦 1040）、1500 回
      学習の経過: 25 回ごと（重い課題は 50 回ごと）に、表との差・値に届いた光を記録する。Adam（0.01）、3 seed。
指標: 経過の曲線、表との差が 0.15（重い課題は 0.85）を切った回、最後の表との差・光、学習で板が動いた量。
結果の置き場: results/exp46_<課題>.txt（1 回 = 1 行、曲線つき）
実行: python experiments/exp46_plate_patterns.py --task easy --device cuda / --task hard --device cuda（experiments/sweeps/sweep46.sh）
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="easy", choices=["easy", "hard"])
ap.add_argument("--cols", type=int, default=20)
ap.add_argument("--dz", type=float, default=64.0)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--plates", default="all")
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2); dev = a.device
out = a.out or f"results/exp46_{a.task}.txt"
if a.task == "easy":
    src = open(os.path.join(os.path.dirname(__file__), "exp21_viewpoint.py"), encoding="utf-8").read().split("\nres = dict(exp=")[0]
    sys.argv = [sys.argv[0]]; ns = {"__file__": __file__}; exec(src, ns)
    OBJ, VAL, VIEW, TABLE = ns["OBJ"], ns["VAL"], ns["VIEW"], ns["TABLE"]
    N, steps, every, thr = 128, 600, 25, 0.15
    slot = {(o, v): 6 * i + VIEW[v] for i, o in enumerate(OBJ) for v in VIEW}; slot.update({x: 30 + j for j, x in enumerate(VAL)})
    pairs = [(o, v) for o in OBJ for v in VIEW]; dist_of = lambda o, v: TABLE[o][v]
else:
    C = json.load(open("results/aozora_adj_nouns.json", encoding="utf-8"))
    SKIP = set("間 方 物 所 中 大 時 長 処 事 小 日 黒 白 面 笑 子 美 奴 調子 気持 話 力 心 言葉 色 音 形 声 様子 上 下 前 後 外 内 他 方々 程 位 者 人 一 二 三".split())
    OBJ = [n for n in C["nouns"] if n not in SKIP][:80]; VO = C["VIEW"]; VAL = [x for v in VO.values() for x in v]; VIEW = {v: i + 1 for i, v in enumerate(VO)}
    N, steps, every, thr = 1040, 1500, 50, 0.85
    SYM = OBJ + VAL; slot = {(o, v): 10 * i + VIEW[v] for i, o in enumerate(OBJ) for v in VIEW}; slot.update({x: 10 * (len(OBJ) + j) for j, x in enumerate(VAL)})
    pairs = [(o, v) for o in OBJ for v in VIEW]
    dist_of = lambda o, v: ({k: c / sum(C["count"][o][v].values()) for k, c in C["count"][o][v].items()} if sum(C["count"][o][v].values()) >= 3 else {x: 1 / len(VO[v]) for x in VO[v]})
L, K = 2 * N, a.cols
yy = torch.arange(N, dtype=torch.float32)[:, None]; cc = torch.arange(K, dtype=torch.float32)[None, :]
H = torch.exp(1j * a.dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2)).to(dev)
wave = lambda k: (torch.exp(1j * 2 * math.pi * k * yy[:, 0] / N) / math.sqrt(N)).to(torch.complex64)
u0 = torch.stack([wave(slot[p]) for p in pairs]).to(dev); vb = torch.tensor([slot[x] for x in VAL], device=dev)
q = torch.zeros(len(pairs), len(VAL) + 1)
for k, (o, v) in enumerate(pairs):
    for x, p in dist_of(o, v).items():
        q[k, VAL.index(x)] = p
q = q.to(dev)
sc = N / 128                                                                               # 縞の周波数は縦の長さに合わせて伸ばす


def stripes(f, g):
    return torch.sin(2 * math.pi * f * sc * yy / N + 2 * math.pi * torch.rand(1, K, generator=g))


def band_noise(g, lo, hi):
    """縦方向に周波数 lo〜hi だけを持つ乱数（列ごとにばらばら）。"""
    Z = torch.fft.fft(torch.randn(N, K, generator=g), dim=0); f = torch.fft.fftfreq(N) * N
    m = ((f.abs() >= lo * sc) & (f.abs() <= hi * sc)).float()[:, None]
    x = torch.fft.ifft(Z * m, dim=0).real
    return x / (x.std() + 1e-9)


lin = lambda f0, f1: f0 + (f1 - f0) * cc / (K - 1)
inv = lambda f0, f1: f0 / (1 + cc * (f0 / f1 - 1) / (K - 1))
PLATES = {"平ら（0）": lambda g: torch.zeros(N, K), "白色雑音 0.1": lambda g: 0.1 * torch.randn(N, K, generator=g),
          "白色雑音 1": lambda g: torch.randn(N, K, generator=g), "白色雑音 ±π": lambda g: math.pi * (2 * torch.rand(N, K, generator=g) - 1),
          "2 値（0 か π）": lambda g: math.pi * (torch.rand(N, K, generator=g) < 0.5).float(),
          "縞 左ほど高い（直線）": lambda g: stripes(lin(40, 2), g), "縞 左ほど高い（反比例）": lambda g: stripes(inv(40, 2), g),
          "縞 左ほど低い（直線）": lambda g: stripes(lin(2, 40), g), "縞 全部の列で 20 周": lambda g: stripes(torch.full((1, K), 20.0), g),
          "縦の低い周波数だけの乱数（1〜8）": lambda g: band_noise(g, 1, 8), "縦の高い周波数だけの乱数（32〜64）": lambda g: band_noise(g, 32, 64),
          "入口側の半分だけ白色雑音 ±π": lambda g: math.pi * (2 * torch.rand(N, K, generator=g) - 1) * (cc < K / 2).float(),
          "出口側の半分だけ白色雑音 ±π": lambda g: math.pi * (2 * torch.rand(N, K, generator=g) - 1) * (cc >= K / 2).float(),
          "レンズ（2 次の位相）+ 雑音 0.1": lambda g: -0.002 / sc ** 2 * (yy - N / 2) ** 2 * torch.ones(1, K) + 0.1 * torch.randn(N, K, generator=g),
          "縦の位置で周波数が上がる（チャープ）": lambda g: torch.sin(2 * math.pi * (yy / N) ** 2 * 40 * sc + 2 * math.pi * torch.rand(1, K, generator=g))}


def run(P, u):
    u = torch.nn.functional.pad(u, (0, L - N))
    for k in range(K):
        u = torch.cat([u[:, :N] * torch.exp(1j * P[:, k]), u[:, N:]], 1)
        u = torch.fft.ifft(torch.fft.fft(u, n=L, dim=-1, norm="ortho") * H, n=L, dim=-1, norm="ortho")
    return torch.fft.fft(u[:, :N], n=N, dim=-1, norm="ortho").abs()[:, vb] ** 2


done = set()
if os.path.exists(out):                                                                   # 途中から再開できるように
    for line in open(out, encoding="utf-8"):
        r = json.loads(line); done.add((r["plate"], r["seed"]))
for name, mk in PLATES.items():
    if a.plates != "all" and name not in a.plates.split(","):
        continue
    for seed in range(a.seeds):
        if (name, seed) in done:
            continue
        t0 = time.time(); g = torch.Generator().manual_seed(seed); P0 = mk(g).float().to(dev)
        P = torch.nn.Parameter(P0.clone()); opt = torch.optim.Adam([P], lr=0.01); curve = []
        for step in range(steps + 1):
            W = run(P, u0)
            if step % every == 0:
                with torch.no_grad():
                    curve.append([step, float(((W / W.sum(1, keepdim=True)) - q[:, :-1]).abs().sum(1).mean() / 2), float(W.sum(1).mean())])
            if step == steps:
                break
            p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
            loss = -(q * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
        hit = next((s for s, tv, _ in curve if tv < thr), None)
        r = dict(exp="46", task=a.task, plate=name, seed=seed, steps=steps, curve=curve, steps_to_thr=hit, thr=thr, tv=curve[-1][1], light=curve[-1][2],
                 move=float((P.detach() - P0).norm() / math.sqrt(P0.numel())), sec=time.time() - t0, device=dev, torch=torch.__version__)
        with open(out, "a", encoding="utf-8") as fo:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{a.task} {name} seed {seed}: {thr} を切った回 {hit}  表との差 {r['tv']:.3f}  光 {100 * r['light']:.1f}%  板の動き {r['move']:.2f}  {r['sec']:.0f} 秒", flush=True)
