"""実験24: 手がかりの数と「物当て」— 観点の掛け算を光でするときの粗さ（量子化・雑音）への強さ

着想: 実験21 で、観点ごとの逆引きの明るさを掛け算すると、手がかりから物を推し量れた。掛け算を光の中でするには、
      (1) 物の点の面で 2 つの光を非線形の結晶で混ぜる（和の周波数の光は、2 つの光の振幅の積 → 明るさは積）、
      (2) 明るさを測って、その値で次の光の通り道の透け具合を変える（測って出し直す）、のどちらかになる。どちらも明るさの掛け算と同じ式になるので、
      ここでは「掛け算の前の明るさが粗い（何ビットで測るか）・雑音がある」ときに、物当てがどれだけ崩れるかを見る（アナログな装置で作るときの目安）。
方法: 実験21 と同じ器を学習（物 5・値 12・観点 5、seed 0〜2）。手がかりの作り方: 物を 1 つ選び、観点を k 個（1〜5）選び、その物の表の割合に従って値を引く
      （手がかり自体が確率的なので、表を知っていても外れることがある）。各条件 2000 問。
      物当て = 掛け算の点数が一番の物。比較相手: 表で計算した点数（ナイーブベイズ、デジタル）、偶然 20%。
      粗さ: 逆引きの明るさを b ビット（2^b 段、最大を 1 として）に丸める、または相対 σ の雑音を掛ける。
結果の置き場: results/exp24.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp24_infer_clues.py
"""
import argparse, json, math, os, sys

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp24.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
src = open(os.path.join(os.path.dirname(__file__), "exp21_viewpoint.py"), encoding="utf-8").read().split("res = dict(")[0]
sys.argv = [sys.argv[0], "--steps", str(a.steps)]
ns = {"__file__": __file__}; exec(src, ns)                                               # 実験21 の器・表・逆引きをそのまま使う
OBJ, VAL, VIEW, TABLE, freq, N, L = (ns[k] for k in ("OBJ", "VAL", "VIEW", "TABLE", "freq", "N", "L"))
res = dict(exp="24", args=vars(a), rows=[])
for seed in range(a.seeds):
    g = torch.Generator().manual_seed(seed)
    A = torch.nn.Parameter(0.1 * torch.randn(4, ns["F"], ns["F"], generator=g)); opt = torch.optim.Adam([A], lr=0.01)
    for step in range(a.steps):
        W = ns["spec"](ns["forward"](A, ns["u0"]))[:, ns["bins"]]
        p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)
        loss = -(ns["q"] * torch.log(p + 1e-9)).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                                                                 # 逆引きの明るさの表 back[(観点, 値)] = 物ごとの明るさ
        back = {}
        for v, d in VIEW.items():
            for val in VAL:
                S = ns["spec"](ns["backward"](A, torch.nn.functional.pad(ns["wave"](freq[val])[None], (0, L - N))))[0]
                back[(v, val)] = np.array([float(S[(freq[o] + d) % N]) for o in OBJ])
    top = max(float(x.max()) for x in back.values())
    rng = np.random.default_rng(seed)
    for k in range(1, 6):
        qs = []
        for _ in range(a.n):
            o = rng.integers(len(OBJ)); views = rng.choice(list(VIEW), k, replace=False)
            clues = [(v, rng.choice(list(TABLE[OBJ[o]][v]), p=list(TABLE[OBJ[o]][v].values()))) for v in views]
            qs.append((o, clues))
        row = dict(seed=seed, k=k)
        for cond in ("表（デジタル）", "光の器", "光の器・8 ビット", "光の器・4 ビット", "光の器・2 ビット", "光の器・雑音 10%", "光の器・雑音 30%"):
            hit = hit_truth = 0
            for o, clues in qs:
                if cond.startswith("表"):
                    sc = np.prod([[TABLE[x][v].get(val, 0) for x in OBJ] for v, val in clues], axis=0)
                else:
                    ms = []
                    for v, val in clues:
                        m = back[(v, val)] / top
                        if "ビット" in cond:
                            b = int(cond.split("・")[1].split()[0]); m = np.round(m * (2 ** b - 1)) / (2 ** b - 1)
                        if "雑音" in cond:
                            m = m * (1 + float(cond.split("雑音 ")[1][:-1]) / 100 * rng.standard_normal(len(OBJ))).clip(min=0)
                        ms.append(m)
                    sc = np.prod(ms, axis=0)
                best = np.flatnonzero(sc == sc.max()); hit += float(o in best) / len(best)        # 同点は等分
            row[cond] = hit / a.n
        res["rows"].append(row)
        print(f"seed {seed} 手がかり {k} 個: " + "  ".join(f"{c} {100 * row[c]:.1f}" for c in row if c not in ("seed", "k")), flush=True)
res.update(numpy=np.__version__, torch=torch.__version__)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
