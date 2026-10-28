"""実験55: 語板（ごばん）を会話データから作る — 名詞 1 語 = 1 枚の光の器、5 つの観点（どんな・を・が・で・に）

着想: 2026-10-28 持ち主「語板作成始めようか」。実験51・52 の語板（src/grain.py）を、手で作った表ではなく、本物の会話から学ぶ最初の一歩。
材料（非公開フォルダ、公開しない）: 実際の雑談（RealPersonaChat、CC BY-SA 4.0）・旅行の相談（JMultiWOZ、CC BY-SA 4.0）・青空文庫の会話文（著作権切れ）を
      Sudachi で語に分け、名詞ごとに 5 つの観点の値を数えたもの（Sin-CNN3_private/tools/extract_views.py）。対話の番号で 9 割を学習、1 割を確かめに使う。
        どんな = 形容詞 + 名詞（美味しい 料理）、を = 名詞 を 動詞（犬 を 飼う）、が = 名詞 が 動詞・形容詞（雨 が 降る）、で = 名詞 で 動詞、に = 名詞 に 動詞
方法: 学習用で 100 回以上出た名詞のうち、形式的な名詞（こと・物・所 …）を除いた上位 400 語。観点ごとの値 = 上位 47 個 + 「その他」。
      語板 1 枚 = 画素の板 8 列（縦 64、src/grain.py と同じ器）。400 枚をまとめて計算し、Adam（0.01）600 回、3 seed。
      観点の数が 3 回未満の (名詞, 観点) は学習に使わない。
確かめ（学習に使っていない 1 割の会話）:
  a. 当たり: 確かめ用の (名詞, 観点, 値) ごとに、値の確率（交差エントロピー）と、上位 1・5 に入るか。
     比べ: 数えた表（学習用の数 + 0.5 で平らに）、名詞を見ない（観点だけの割合）。
  b. 逆引き: 値（例: を → 飼う）の光を逆向きに通したときに明るい名詞の上位（逆引きの明るさ = 順引きの明るさ）。
  c. ○○ 当て: 確かめ用の文で名詞を 1 つ隠し、その名詞の観点の手がかり（隣の動詞・形容詞）の掛け算で 400 語から当てる（実験52 の形）。
     比べ: 数えた表の掛け算、よく出る順。
結果の置き場: results/exp55.txt（数字と、少しの例語だけ。会話の文そのものは置かない）
実行: CUDA_VISIBLE_DEVICES= python experiments/exp55_goban_corpus.py
"""
import argparse, collections, json, math, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grain import N, K, DZ, view_wave

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="D:/AI_git/Sin-CNN3_private/data/views")
ap.add_argument("--nouns", type=int, default=400)
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--device", default="cpu")
ap.add_argument("--out", default="results/exp55.txt")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2); dev = a.device
VIEWS = ["どんな", "を", "が", "で", "に"]; NV = 48                                       # 観点ごとの値 47 個 + その他（縦 64 の器に入る上限近く）
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
C = json.load(open(f"{a.data}/counts.json", encoding="utf-8"))
tot = {n: sum(sum(c.values()) for c in vs.values()) for n, vs in C["train"].items()}
NOUNS = [n for n in sorted(tot, key=tot.get, reverse=True) if n not in SKIP][:a.nouns]
allv = {v: collections.Counter() for v in VIEWS}
for n, vs in C["train"].items():
    for v, c in vs.items():
        allv[v].update(c)
VALS = {v: [x for x, _ in allv[v].most_common(NV - 1)] + ["その他"] for v in VIEWS}
vidx = lambda v, x: VALS[v].index(x) if x in VALS[v][:-1] else NV - 1


def table(split, n):
    t = torch.zeros(len(VIEWS), NV)
    for i, v in enumerate(VIEWS):
        for x, c in C[split].get(n, {}).get(v, {}).items():
            t[i, vidx(v, x)] += c
    return t


TR = torch.stack([table("train", n) for n in NOUNS]); TE = torch.stack([table("test", n) for n in NOUNS])   # (語, 観点, 値) の数
use = TR.sum(2) >= 3; tgt = TR / TR.sum(2, keepdim=True).clamp(min=1)
prior = torch.stack([torch.tensor([allv[v][x] for x in VALS[v][:-1]] + [sum(allv[v].values()) - sum(allv[v][x] for x in VALS[v][:-1])], dtype=torch.float32) for v in VIEWS])
prior = prior / prior.sum(1, keepdim=True)                                                  # 名詞を見ない（観点だけ）
L = 2 * N; H = torch.exp(1j * DZ * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2)).to(dev)
u0 = view_wave(range(1, len(VIEWS) + 1)).to(dev); VAL0 = 10


def run(P):
    """P: (語, N, K) → (語, 観点, 値) の明るさ。全部の語板をまとめて計算する。"""
    u = torch.nn.functional.pad(u0, (0, L - N))[None].expand(P.shape[0], -1, -1)
    for k in range(K):
        u = torch.cat([u[..., :N] * torch.exp(1j * P[:, None, :, k]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
    return torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs()[..., VAL0:VAL0 + NV] ** 2


def evaluate(Pr):
    """Pr: (語, 観点, 値) の確率。確かめ用の数で重みをつけた交差エントロピーと上位 1・5。"""
    Pr = Pr.cpu(); w = TE; lp = torch.log2(Pr.clamp(min=1e-9))
    bits = float(-(w * lp).sum() / w.sum())
    rank = (Pr[..., None, :] > Pr[..., :, None]).sum(-1)                                     # その値より確率の高い値の数
    top1 = float((w * (rank < 1)).sum() / w.sum()); top5 = float((w * (rank < 5)).sum() / w.sum())
    return dict(bits=bits, top1=top1, top5=top5)


res = []
cnt = (TR + 0.5) / (TR + 0.5).sum(2, keepdim=True)                                         # 数えた表（+0.5 で平らに）
base = {"数えた表": evaluate(cnt), "名詞を見ない": evaluate(prior[None].expand(len(NOUNS), -1, -1))}
print("比べ:", {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in base.items()}, flush=True)
# ○○ 当ての文（確かめ用）: 隠す名詞が 400 語に入っていて、手がかりが 1 つ以上あるもの
cloze = []
for line in open(f"{a.data}/sentences_test.jsonl", encoding="utf-8"):
    d = json.loads(line); by = collections.defaultdict(list)
    for i, v, x in d["hits"]:
        by[i].append((v, x))
    for i, cl in by.items():
        n = d["toks"][i][1]
        if n in NOUNS:
            cloze.append((NOUNS.index(n), [(VIEWS.index(v), vidx(v, x)) for v, x in cl]))
freq = torch.tensor([tot[n] for n in NOUNS], dtype=torch.float32)


def cloze_eval(Pr):
    Pr = Pr.cpu(); lp = torch.log(Pr.clamp(min=1e-6)); r1 = r10 = 0; ranks = []
    for ans, cl in cloze:
        sc = sum(lp[:, v, x] for v, x in cl) + torch.log(freq)                             # 手がかりの掛け算 × よく出る度合い（事前の割合）
        rk = int((sc > sc[ans]).sum()); ranks.append(rk); r1 += rk < 1; r10 += rk < 10
    return dict(top1=r1 / len(cloze), top10=r10 / len(cloze), n=len(cloze))


fr = freq / freq.sum(); base_cloze = {"数えた表": cloze_eval(cnt), "よく出る順だけ": dict(top1=float(fr.max() == fr[cloze[0][0]]), top10=0, n=len(cloze))}
rk = torch.argsort(fr, descending=True); pos = {int(j): r for r, j in enumerate(rk)}
base_cloze["よく出る順だけ"] = dict(top1=sum(pos[c[0]] < 1 for c in cloze) / len(cloze), top10=sum(pos[c[0]] < 10 for c in cloze) / len(cloze), n=len(cloze))
print("○○ 当て 比べ:", base_cloze, flush=True)
for seed in range(a.seeds):
    t0 = time.time(); g = torch.Generator().manual_seed(seed)
    P = torch.nn.Parameter((math.pi * (torch.rand(len(NOUNS), N, K, generator=g) < 0.5).float()).to(dev))   # 2 値の初期板（学び 45）
    opt = torch.optim.Adam([P], lr=0.01); m = use.to(dev); q = tgt.to(dev)
    for step in range(a.steps):
        W = run(P); p = torch.cat([W, (1 - W.sum(-1, keepdim=True)).clamp(min=1e-9)], -1)
        qq = torch.cat([q, torch.zeros(*q.shape[:2], 1, device=dev)], -1)
        loss = -(qq * torch.log(p + 1e-9)).sum(-1)[m].mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        W = run(P); light = float(W.sum(-1)[m].mean())
        Pr = 0.98 * W / W.sum(-1, keepdim=True) + 0.02 / NV                                  # 割合に直し、少しだけ平らに（0 の値を避ける）
    ev = evaluate(Pr); cz = cloze_eval(Pr)
    # 逆引き: 値の光で明るい名詞（順引きの明るさ = 逆引きの明るさ）
    rev = {}
    for v, x in [("を", "飼う"), ("が", "降る"), ("を", "食べる"), ("どんな", "美味しい"), ("を", "見る"), ("に", "乗る"), ("で", "買う")]:
        if x in VALS[v]:
            b = W[:, VIEWS.index(v), VALS[v].index(x)].cpu(); rev[f"{v} {x}"] = [NOUNS[int(j)] for j in torch.argsort(b, descending=True)[:6]]
    row = dict(exp="55", seed=seed, nouns=len(NOUNS), steps=a.steps, eval=ev, base=base, cloze=cz, base_cloze=base_cloze, light=light, reverse=rev,
               train_rows=int(use.sum()), sec=time.time() - t0, device=dev, torch=torch.__version__)
    res.append(row)
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(seed, "語板:", {k: round(v, 3) for k, v in ev.items()}, "○○ 当て", {k: round(v, 3) for k, v in cz.items()}, f"光 {100 * light:.0f}%  {row['sec']:.0f} 秒", flush=True)
    print("  逆引き:", rev, flush=True)
