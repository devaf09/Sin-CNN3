"""実験99: 語板 v5 — 全品詞（名詞・動詞・形容詞・形状詞）に意味の観点と品詞の観点を持たせる（持ち主の依頼 2026-11-02）

着想: 持ち主「品詞とそれ以外の属性の関連をみつけたい」「質問に対し何に重みを追加するか、それとバッファ板の変動にも使いたい」。
      語板 v4 は名詞だけなので、品詞が揺れず、品詞と意味の関連が見えなかった。動詞・形容詞にも意味の観点を持たせ、品詞の観点も一緒に入れる。
観点（周波数の番号 = 全部の語で共通。語ごとに、その品詞の観点だけを使う）:
  名詞の観点 9 本（周波数 1〜9）: どんな・を・が・で・に・の前・の後・と・は。値 = 語板 v3 と同じ上位 45（+ その他）。
  動詞の観点 5 本（周波数 10〜14）: が・を・に・で・と に来る名詞の仲間（実験77 の 100 組）。例: 食べる の を → 食べ物の仲間。
  形容詞・形状詞の観点 2 本（周波数 15・16）: どんな名詞に付くか（〜な X・〜い X）・何が〜か（X は 〜）。値 = 名詞の仲間 100。
  品詞の観点 8 本（周波数 17〜24）: 実験78 の D 案（大分類 14 + 中分類）。全部の語。
  数は語板 v4 と同じ材料（merged2.pkl: 名詞 → 観点 → 値）を、動詞・形容詞の側から数え直したもの（名詞 n の「を 食べる」= 食べる の「を n の仲間」）。
語: 名詞 2 万（語板 v3 の多い順）・動詞 3000・形容詞と形状詞 2000（材料の多い順）。
器: 縦 256 × 板 24（観点が多いので板を増やす、実験79）。出口 = 値 r を (r + 7 × 観点の番号) mod 101 に、その他は 101 + 観点の番号に（出口のずらし）。500 回。
結果: results/exp99.txt。語板: 非公開フォルダの models/goban_v5_<group>.pt
実行: python experiments/exp99_goban_v5.py --device cuda --group noun（experiments/sweeps/sweep99_*.sh）
"""
import argparse, collections, json, math, os, pickle, sys, time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--group", default="noun")                                                 # noun / verb / adj
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import _h, init_plates, VAL0, prob

sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time(); dev = a.device; PRIV = "D:/AI_git/Sin-CNN3_private"
NV9 = ["どんな", "を", "が", "で", "に", "の前", "の後", "と", "は"]
VV = ["V:が", "V:を", "V:に", "V:で", "V:と"]; AV = ["A:どんな", "A:は"]
POSV = ["品詞", "P:名詞", "P:動詞", "P:形容詞", "P:形状詞", "P:感動詞", "P:接尾辞", "P:助詞"]
ALLV = NV9 + VV + AV + POSV; FREQ_OF = {v: i + 1 for i, v in enumerate(ALLV)}               # 観点 → 入口の周波数（1 から）
NS = 101; NOUT = NS + len(ALLV)                                                              # 出口: 値の場所 101 + 観点ごとのその他
SLOT = lambda j, r: (r + 7 * j) % NS
D = pickle.load(open(f"{PRIV}/data/views/merged2.pkl", "rb"))
NX = torch.load(f"{PRIV}/models/goban_next_web2.pt", weights_only=False)                     # 名詞の仲間 100（語板 v3 の 3 万語）
CL = {w: int(c) for w, c in zip(NX["words"], NX["cluster"])}
PM = torch.load(f"{PRIV}/models/goban_pos.pt", weights_only=False); PWI = {w: i for i, w in enumerate(PM["words"])}
SKIP = set("こと 物 所 方 事 為 訳 時 中 前 後 内 外 上 下 他 気 感じ 一緒 自分 仕方 様 辺 程 頃 間 度 位 際 通り 儘 序で 筈 振り 積もり 以上 以外 場合 本当 皆 皆さん 今日 今 昨日 明日 先 次 横 側".split())
ALLN = [n for n in sorted(D["noun_total"], key=D["noun_total"].get, reverse=True) if n not in SKIP][:30000]
# 名詞の観点の値（語板 v3 と同じ上位 45）
allv = {v: collections.Counter() for v in NV9}
for n in ALLN:
    for v in NV9:
        allv[v].update(D["train"][n].get(v, {}))
VALS = {v: [x for x, _ in allv[v].most_common(45)] for v in NV9}
VALS.update({v: list(range(100)) for v in VV + AV})                                          # 動詞・形容詞の観点の値 = 名詞の仲間の番号
# 動詞・形容詞の側から数え直す（名詞 n の「観点 v の値 x」= x の「V:v / A:… の値 n の仲間」）
inv = {s: collections.defaultdict(lambda: collections.defaultdict(collections.Counter)) for s in ("train", "test")}
for s in inv:
    for n in ALLN:
        c = CL.get(n)
        if c is None:
            continue
        for v, vals in D[s].get(n, {}).items():
            for x, m in vals.items():
                if v in ("が", "を", "に", "で", "と"):
                    inv[s][x]["V:" + v][c] += m
                elif v == "どんな":
                    inv[s][x]["A:どんな"][c] += m
                elif v == "は":
                    inv[s][x]["A:は"][c] += m
POSD = lambda w: PM["table"][PWI[w], 0, :14] if w in PWI else None                           # 品詞の大分類の割合
if a.group == "noun":
    WORDS = ALLN[:20000]; VIEWS = NV9 + POSV
elif a.group == "verb":
    tot = collections.Counter({x: sum(sum(c.values()) for k, c in vs.items() if k.startswith("V:")) for x, vs in inv["train"].items()})
    WORDS = [x for x, _ in tot.most_common() if POSD(x) is not None and PM["top"][int(POSD(x).argmax())] == "動詞"][:3000]; VIEWS = VV + POSV
else:
    tot = collections.Counter({x: sum(sum(c.values()) for k, c in vs.items() if k.startswith("A:")) for x, vs in inv["train"].items()})
    WORDS = [x for x, _ in tot.most_common() if POSD(x) is not None and PM["top"][int(POSD(x).argmax())] in ("形容詞", "形状詞")][:2000]; VIEWS = AV + POSV
if a.quick:
    WORDS = WORDS[:200]
print(a.group, "語", len(WORDS), "観点", len(VIEWS), f"{time.time() - t0:.0f} 秒", flush=True)
VJ = [ALLV.index(v) for v in VIEWS]                                                          # 観点の全体での番号（出口のずらしと入口の周波数に使う）


def table(split):
    """(語, 観点, 出口) の数（意味の観点）。品詞の観点は実験78 の割合を数 100 として入れる。"""
    T = torch.zeros(len(WORDS), len(VIEWS), NOUT); use = torch.zeros(len(WORDS), len(VIEWS), dtype=torch.bool)
    for i, w in enumerate(WORDS):
        for k, v in enumerate(VIEWS):
            j = VJ[k]
            if v in NV9:
                for x, m in D[split].get(w, {}).get(v, {}).items():
                    r = VALS[v].index(x) if x in POS45.get(v, {}) else None
                    T[i, k, SLOT(j, POS45[v][x]) if r is not None else NS + j] += m
            elif v in VV + AV:
                for c, m in inv[split].get(w, {}).get(v, {}).items():
                    T[i, k, SLOT(j, c)] += m
            else:
                if w in PWI and split == "train":
                    pv = PM["views"].index(v.replace("P:", "")) if v != "品詞" else 0
                    t = PM["table"][PWI[w], pv]
                    if float(t[:-1].sum()) > 0.5:
                        for r in range(len(t) - 1):
                            T[i, k, SLOT(j, r)] += 100 * float(t[r])
                        T[i, k, NS + j] += 100 * float(t[-1])
            use[i, k] = T[i, k].sum() > 0
    return T, use


POS45 = {v: {x: r for r, x in enumerate(VALS[v])} for v in NV9}
TR, USE = table("train"); TE, _ = table("test")
print("表", f"{time.time() - t0:.0f} 秒", flush=True)
mask_slots = torch.zeros(len(VIEWS), NOUT, dtype=torch.bool)                                  # 観点ごとに使う出口（値の場所 + その観点のその他）
for k, v in enumerate(VIEWS):
    nv = len(VALS[v]) if v in VALS else 14
    for r in range(nv):
        mask_slots[k, SLOT(VJ[k], r)] = True
    mask_slots[k, NS + VJ[k]] = True
pri = (TR.sum(0) + 0.5 * mask_slots) / (TR.sum(0) + 0.5 * mask_slots).sum(1, keepdim=True)
tg = (TR + 20 * pri[None]) / (TR.sum(2, keepdim=True) + 20) * mask_slots[None]
N, K = 256, 24; H = _h(dev, N); yy = torch.arange(N, dtype=torch.float32, device=dev)
WAVES = torch.stack([torch.exp(1j * 2 * math.pi * FREQ_OF[v] * yy / N) / math.sqrt(N) for v in VIEWS]).to(torch.complex64)   # 観点ごとの入口の波（周波数は全部の語で共通）


def run_v(P):
    """板 P（語, 縦, 板）に、この組の観点の波だけを通し、(語, 観点, 出口) の明るさを返す。"""
    u = torch.nn.functional.pad(WAVES, (0, N))[None].expand(P.shape[0], -1, -1)
    for k in range(K):
        u = torch.cat([u[..., :N] * torch.exp(1j * P[:, None, :, k]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
    return torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs()[..., VAL0:VAL0 + NOUT] ** 2


B = 300; Ps = []
for i in range(0, len(WORDS), B):
    P = init_plates(len(WORDS[i:i + B]), i, dev, N, K).requires_grad_(True); opt = torch.optim.Adam([P], lr=0.03)
    q = tg[i:i + B].to(dev); m = USE[i:i + B].to(dev)
    for st in range(a.steps):
        W = run_v(P); p = torch.cat([W, (1 - W.sum(-1, keepdim=True)).clamp(min=1e-9)], -1)
        loss = -(torch.cat([q, torch.zeros(*q.shape[:2], 1, device=dev)], -1) * torch.log(p + 1e-9)).sum(-1)[m].mean()
        opt.zero_grad(); loss.backward(); opt.step()
    Ps.append(P.detach().cpu())
    if i % 3000 == 0:
        print(i, "/", len(WORDS), round(float(loss), 4), f"{time.time() - t0:.0f} 秒", flush=True)
P = torch.cat(Ps)
with torch.no_grad():
    W = torch.cat([run_v(P[i:i + B].to(dev)).cpu() for i in range(0, len(WORDS), B)])
Wm = W * mask_slots[None]; pb = Wm / Wm.sum(-1, keepdim=True).clamp(min=1e-12); pb = 0.98 * pb + 0.02 * mask_slots[None] / mask_slots.sum(1)[None, :, None]
sem = torch.tensor([v in NV9 + VV + AV for v in VIEWS])
w_te = TE[:, sem]; bits = lambda Pr: float(-(w_te * torch.log2(Pr[:, sem].clamp(min=1e-9))).sum() / w_te.sum().clamp(min=1))
res = dict(exp="99", group=a.group, words=len(WORDS), views=VIEWS, board_bits=bits(pb), table_bits=bits(tg + 1e-9 * mask_slots[None]), sec=time.time() - t0, device=dev)
posk = [k for k, v in enumerate(VIEWS) if v == "品詞"][0]
tgp = tg[:, posk]; pbp = pb[:, posk]
res["pos_top1_agree"] = float((pbp.argmax(1) == tgp.argmax(1))[USE[:, posk]].float().mean())
torch.save(dict(P=P, words=WORDS, views=VIEWS, all_views=ALLV, vals={v: VALS[v] for v in VIEWS if v in VALS}, slot_ns=NS, group=a.group,
                pos_top=PM["top"], pos_sub=PM["sub"]), f"{PRIV}/models/goban_v5_{a.group}{'_quick' if a.quick else ''}.pt")
with open("results/exp99.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print(res)
