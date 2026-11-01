"""実験92: 役割フィルター — 文節の語の並び（と係る先）から、その文節の役割（係り方）を当てる小さな光の器（持ち主の選んだ 1 番）

着想: 2026-11-02 持ち主「疑問 → 役割 → 否定はいい関連する流れ。これをすべての品詞で」。実験87〜89 では文節の役割を手の決まり（role、最後の助詞など）で決めていた。
      これを、係り受け（GiNZA）を教え役にして学ぶ小さなフィルターに置き換える。実験91 の「求めるもの」の通り、助詞・助動詞と品詞の並びが手がかり。
材料: 非公開フォルダの tools/gen_bunsetsu_roles.py が作った 22 万文節（3 万文）。教え役のラベル = GiNZA の係り方（よく出る 13 + その他）。文ごとに 9:1。
入力の光（特徴ごとに 1 本の周波数）: 文節の中の助詞・助動詞の語（よく出る 60）、文節の最初と最後の語の品詞（大分類・中分類）、文の根か、
      係る先の文節の最初の語の品詞と、係る先の中の助詞・助動詞の語。
出力: 係り方の割合。器 = 縦 512 × 板 16、参照の光あり、3000 回（CPU）。
比べ: 表（最後の助詞 + 最初の品詞 → 係り方）、1 層の重み（光なし）、手の決まり（実験87 の role を係り方に読み替えたもの: が・は → nsubj、を → obj、に・へ・で・から → obl、どんな → acl、の → nmod）。
結果: results/exp92.txt。実行: python experiments/exp92_role_filter.py
"""
import collections, json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import init_plates, _h, VAL0

sys.stdout.reconfigure(encoding="utf-8"); torch.manual_seed(0); PRIV = "D:/AI_git/Sin-CNN3_private"
D = json.load(open(f"{PRIV}/data/views/bunsetsu_roles.json", encoding="utf-8"))
lab_c = collections.Counter(d["dep"] for d in D if d["split"] == "train")
LABS = [l for l, _ in lab_c.most_common(13)] + ["その他"]; LI = {l: i for i, l in enumerate(LABS)}; NL = len(LABS)
fn = lambda w: w[2].startswith(("助詞", "助動詞"))
fwc = collections.Counter(w[0] for d in D if d["split"] == "train" for w in d["words"] if fn(w))
FW = [w for w, _ in fwc.most_common(60)]
pos2 = lambda t: "-".join(t.split("-")[:2])
posc = collections.Counter(pos2(w[2]) for d in D[:50000] for w in d["words"])
POSL = [p for p, _ in posc.most_common(40)]
FEATS = (["fw:" + w for w in FW] + ["first:" + p for p in POSL] + ["last:" + p for p in POSL] + ["root"]
         + ["hfirst:" + p for p in POSL] + ["hfw:" + w for w in FW])
FI = {f: i for i, f in enumerate(FEATS)}; NF = len(FEATS)


def feats(d):
    ws = d["words"]; f = ["fw:" + w[0] for w in ws if fn(w)]
    cont = [w for w in ws if not fn(w) and not w[2].startswith("補助記号")] or ws
    f += ["first:" + pos2(cont[0][2]), "last:" + pos2(ws[-1][2])]
    if d["is_root"]:
        f.append("root")
    else:
        hc = [w for w in d["head_words"] if not fn(w) and not w[2].startswith("補助記号")] or d["head_words"]
        if hc:
            f.append("hfirst:" + pos2(hc[0][2]))
        f += ["hfw:" + w[0] for w in d["head_words"] if fn(w)]
    return [FI[x] for x in f if x in FI]


def tens(rows):
    X = torch.zeros(len(rows), NF); y = torch.tensor([LI.get(d["dep"], NL - 1) for d in rows])
    for i, d in enumerate(rows):
        X[i, feats(d)] = 1.0
    return X, y


TR = [d for d in D if d["split"] == "train"]; TE = [d for d in D if d["split"] == "test"]
Xtr, ytr = tens(TR); Xte, yte = tens(TE)
print("学習", len(TR), "確かめ", len(TE), "特徴", NF, "係り方", NL, flush=True)
acc = lambda P: float((P.argmax(1) == yte).float().mean())
res = dict(exp="92", train=len(TR), test=len(TE), labels=LABS)
# 手の決まり（実験87 の role を係り方に読み替え）
spec = os.path.join(os.path.dirname(__file__), "exp87_story_tree_qa.py")
src = open(spec, encoding="utf-8").read(); i0 = src.index("def role(b):"); i1 = src.index("\ndef build(S):")
ns = {"CONTENT": ("名詞", "動詞", "形容詞", "形状詞", "代名詞")}; exec(src[i0:i1], ns)
MAP = {"が": "nsubj", "は": "nsubj", "を": "obj", "に": "obl", "へ": "obl", "で": "obl", "から": "obl", "どんな": "acl", "の": "nmod", "という": "nmod", "引用": "advcl", "も": "nsubj", "と": "obl"}
hand = []
for d in TE:
    b = dict(words=[w[:3] for w in d["words"]], text="".join(w[0] for w in d["words"]), dep="ROOT" if d["is_root"] else "x")
    r = "ROOT" if d["is_root"] else MAP.get(ns["role"](b), "advmod")
    hand.append(LI.get(r, NL - 1))
res["hand"] = float((torch.tensor(hand) == yte).float().mean())
# 表
tab = collections.defaultdict(collections.Counter); glob = collections.Counter()
key = lambda d: ((["root"] if d["is_root"] else []) + [w[0] for w in d["words"] if fn(w)][-1:] + [pos2(([w for w in d["words"] if not fn(w)] or d["words"])[0][2])])
for d in TR:
    tab[tuple(key(d))][LI.get(d["dep"], NL - 1)] += 1; glob[LI.get(d["dep"], NL - 1)] += 1
Pt = torch.zeros(len(TE), NL)
for i, d in enumerate(TE):
    for l, m in (tab.get(tuple(key(d))) or glob).items():
        Pt[i, l] = m
res["table"] = acc(Pt)
# 1 層の重み
Wl = torch.zeros(NF, NL, requires_grad=True); bl = torch.zeros(NL, requires_grad=True); opt = torch.optim.Adam([Wl, bl], lr=0.05)
for st in range(2000):
    idx = torch.randint(0, len(Xtr), (1024,)); loss = torch.nn.functional.cross_entropy(Xtr[idx] @ Wl + bl, ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    res["linear"] = acc(Xte @ Wl + bl)
# 光のフィルター
N, K = 512, 16; H = _h("cpu", N); yy = torch.arange(N, dtype=torch.float32)
WAVE = torch.stack([torch.exp(1j * 2 * math.pi * (k + 1) * yy / N) / math.sqrt(N) for k in range(NF + 1)]).to(torch.complex64)


def forward(P, X):
    u = X.to(torch.complex64) @ WAVE[:NF]; u = u / u.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-9)
    u = torch.nn.functional.pad((u + WAVE[NF][None]) / math.sqrt(2), (0, N))
    for k in range(K):
        u = torch.cat([u[..., :N] * torch.exp(1j * P[:, k]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
    o = torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs()[:, VAL0:VAL0 + NL] ** 2
    return o / o.sum(-1, keepdim=True).clamp(min=1e-12)


P = init_plates(1, 0, "cpu", N, K)[0].requires_grad_(True); STEPS = 3000
opt = torch.optim.Adam([P], lr=0.02); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
for st in range(STEPS):
    idx = torch.randint(0, len(Xtr), (256,)); o = forward(P, Xtr[idx])
    loss = -torch.log(o.gather(1, ytr[idx][:, None])[:, 0] + 1e-9).mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if st % 500 == 0:
        print(st, "回", round(float(loss), 4), flush=True)
with torch.no_grad():
    Po = torch.cat([forward(P, Xte[i:i + 4000]) for i in range(0, len(Xte), 4000)])
res["light"] = acc(Po)
cm = collections.Counter((LABS[int(a)], LABS[int(b)]) for a, b in zip(yte, Po.argmax(1)) if a != b)
res["light_by_label"] = {l: float((Po.argmax(1)[yte == i] == i).float().mean()) for i, l in enumerate(LABS) if (yte == i).any()}
res["confusions"] = cm.most_common(8)
torch.save(dict(P=P.detach(), feats=FEATS, labels=LABS), f"{PRIV}/models/role_filter.pt")
with open("results/exp92.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
print({k: v for k, v in res.items() if k not in ("labels",)})
