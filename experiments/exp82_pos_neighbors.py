"""実験82: 前後の品詞 — 文章から品詞は自然に出てくるか。前・後の品詞の観点を持つ語板で、正引き（次の品詞を当てる）と逆引き（文の型を並べる）

着想: 2026-10-31 持ち主「自然と文章に合わせて体系づけられるなら、構文解析も同じ手が使える。問題は、逆引きすれば構文生成に使える仕組みとするか、
      前も後ろも品詞を想定できる仕組みにするか」。値の仲間 45（実験80）が、名詞との組み合わせだけで意味のまとまりになったのと同じ手を使う。
1. 自然に品詞が出るか: よく出る 5000 語を、すぐ前の語・すぐ後の語（よく出る 300 語）の数の似かたで 40 の仲間に分ける（k 平均法）。
   辞書（Sudachi）の品詞とどれだけ一致するか（仲間ごとにいちばん多い品詞の割合 = 純度）。比べ: でたらめに分けたとき。
2. 前・後の品詞の観点を持つ語板: よく出る 2 万語。観点 1 = すぐ前の語の品詞、観点 2 = すぐ後の語の品詞。値 = 品詞（大分類-中分類、約 40）＋ 助詞・助動詞は語そのもの（よく出る 40）。
   器 = 縦 128 × 板 16、500 回（CPU）。出口は観点ごとにずらす（実験78 の直し）。
   正引き: 確かめ用の材料で、次の語の品詞の当たり（交差エントロピー・上位 1）。表と比べる。
   逆引き: 「前 = を」「前 = が」「後 = た」などを逆に入れ、反応する語（その位置に来られる語）を出す。
   文の型: 「名詞」から始めて、後の観点でいちばん明るい品詞を選び、その品詞の語から次へ…と 6 つ並べる（正引きの連鎖）。
材料: 会話（RealPersonaChat・JMultiWOZ）・青空文庫 300 作品・Web 小説 3000 話（実験だけ、数だけ）。学習用と確かめ用は文書ごとに 9:1。
結果: results/exp82.txt、docs/pos_neighbors_82.md。実行: CUDA_VISIBLE_DEVICES= python experiments/exp82_pos_neighbors.py
"""
import collections, json, math, os, random, re, sys, time, zlib

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t0 = time.time(); PRIV = "D:/AI_git/Sin-CNN3_private"
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").tokenizer(); M = SplitMode.C
KEEP = {"助詞", "助動詞"}


def lab(p, w):
    """品詞の値: 助詞・助動詞は語そのもの（品詞つき）、それ以外は大分類-中分類。"""
    if p[0] in KEEP:
        return f"{p[0]}:{w}"
    return p[0] + ("-" + p[1] if p[1] != "*" else "")


SEQ = {"train": [], "test": []}                                                             # 文ごとの (語, 品詞の値, 大分類)


def add(text, did):
    s = "test" if zlib.crc32(did.encode()) % 10 == 0 else "train"
    for sent in re.split(r"(?<=[。！？\n])", text):
        sent = sent.strip()
        if 2 < len(sent) <= 300:
            SEQ[s].append([(m.normalized_form(), lab(m.part_of_speech(), m.normalized_form()), m.part_of_speech()[0]) for m in tok.tokenize(sent, M)])


for l in open(f"{PRIV}/data/text/conv_all.jsonl", encoding="utf-8"):
    d = json.loads(l)
    if not d["id"].startswith("aozora"):
        add(d["text"], "conv:" + d["id"])
A = [json.loads(l) for l in open(f"{PRIV}/data/text/aozora_full.jsonl", encoding="utf-8")]; random.Random(0).shuffle(A)
for d in A[:300]:
    add(d["text"], "aozora:" + d["id"])
del A
W = [json.loads(l) for l in open(f"{PRIV}/data/text/webnovels2.jsonl", encoding="utf-8")]; random.Random(0).shuffle(W)
for d in W[:3000]:
    add(d["text"], "web:" + d["work"])
del W
print("文", len(SEQ["train"]), len(SEQ["test"]), f"{time.time() - t0:.0f} 秒", flush=True)
res = dict(exp="82")
# ---- 1. 自然に品詞が出るか ----
wc = collections.Counter(w for s in SEQ["train"] for w, _, _ in s)
POS1 = collections.defaultdict(collections.Counter)
for s in SEQ["train"]:
    for w, _, b in s:
        POS1[w][b] += 1
TOPW = [w for w, _ in wc.most_common(5000)]; TI = {w: i for i, w in enumerate(TOPW)}
CTX = [w for w, _ in wc.most_common(300)]; CI = {w: i for i, w in enumerate(CTX)}
X = torch.zeros(len(TOPW), 600)
for s in SEQ["train"]:
    for k, (w, _, _) in enumerate(s):
        i = TI.get(w)
        if i is None:
            continue
        if k > 0 and s[k - 1][0] in CI:
            X[i, CI[s[k - 1][0]]] += 1
        if k + 1 < len(s) and s[k + 1][0] in CI:
            X[i, 300 + CI[s[k + 1][0]]] += 1
Xn = torch.nn.functional.normalize(X.sqrt(), dim=1)
g = torch.Generator().manual_seed(0); C = Xn[torch.randperm(len(TOPW), generator=g)[:40]].clone()
for it in range(30):
    labs = (Xn @ C.T).argmax(1)
    for c in range(40):
        if (labs == c).any():
            C[c] = torch.nn.functional.normalize(Xn[labs == c].mean(0), dim=0)
gold = [POS1[w].most_common(1)[0][0] for w in TOPW]


def purity(L):
    tot = 0
    for c in set(L.tolist()):
        tot += collections.Counter(gold[i] for i in range(len(TOPW)) if int(L[i]) == c).most_common(1)[0][1]
    return tot / len(TOPW)


res["purity"] = purity(labs); res["purity_random"] = purity(torch.randint(0, 40, (len(TOPW),), generator=g))
clusters = []
for c in range(40):
    mem = [TOPW[i] for i in range(len(TOPW)) if int(labs[i]) == c]
    if mem:
        pc = collections.Counter(gold[TI[w]] for w in mem).most_common(2)
        clusters.append((len(mem), pc, mem[:12]))
clusters.sort(key=lambda t: -t[0])
print("純度", res["purity"], "でたらめ", res["purity_random"], f"{time.time() - t0:.0f} 秒", flush=True)
# ---- 2. 前・後の品詞の観点を持つ語板 ----
lc = collections.Counter(l for s in SEQ["train"] for _, l, _ in s)
VALS = [l for l, _ in lc.most_common(80)]; VI = {l: i for i, l in enumerate(VALS)}; NV = len(VALS)   # 品詞の値 80（大分類-中分類 + よく出る助詞・助動詞）
WORDS = [w for w, _ in wc.most_common(20000)]; WI = {w: i for i, w in enumerate(WORDS)}


def table(split):
    T = torch.zeros(len(WORDS), 2, NV + 1)
    for s in SEQ[split]:
        for k, (w, _, _) in enumerate(s):
            i = WI.get(w)
            if i is None:
                continue
            T[i, 0, VI.get(s[k - 1][1], NV) if k > 0 else NV] += 1                          # 前の品詞（文の始めは「その他」）
            T[i, 1, VI.get(s[k + 1][1], NV) if k + 1 < len(s) else NV] += 1                 # 後の品詞（文の終わりは「その他」）
    return T


TR, TE = table("train"), table("test")
pri = (TR.sum(0) + 0.5) / (TR.sum(0) + 0.5).sum(1, keepdim=True); tg = (TR + 5 * pri[None]) / (TR.sum(2, keepdim=True) + 5)
NOUT = NV + 2; PHYS = torch.tensor([[(r + 7 * j) % NV for r in range(NV)] + [NV + j] for j in range(2)])   # 出口のずらし（観点ごと）
to_phys = lambda T: torch.stack([torch.zeros(T.shape[0], NOUT).index_copy(1, PHYS[j], T[:, j]) for j in range(2)], 1)
B = 2000
P = torch.cat([train(to_phys(tg[i:i + B]), torch.ones(len(tg[i:i + B]), 2, dtype=torch.bool), 500, i, lr=0.03, rows=128, plates=16) for i in range(0, len(WORDS), B)])
Wt = torch.cat([run(P[i:i + B], 2, NOUT) for i in range(0, len(WORDS), B)])
pb = prob(torch.stack([Wt[:, j, PHYS[j]] for j in range(2)], 1))
bits = lambda Pr, C: float(-(C * torch.log2(Pr.clamp(min=1e-9))).sum() / C.sum())
top1 = lambda Pr, C: float(C.gather(-1, Pr.argmax(-1, keepdim=True)).sum() / C.sum())
res.update(board_bits_next=bits(pb[:, 1], TE[:, 1]), table_bits_next=bits(tg[:, 1], TE[:, 1]), board_top1_next=top1(pb[:, 1], TE[:, 1]), table_top1_next=top1(tg[:, 1], TE[:, 1]),
           board_bits_prev=bits(pb[:, 0], TE[:, 0]), table_bits_prev=bits(tg[:, 0], TE[:, 0]))
print("次の品詞の当たり 語板", res["board_bits_next"], "表", res["table_bits_next"], f"{time.time() - t0:.0f} 秒", flush=True)
# 逆引き: その位置に来られる語（その値の明るさ × log(語の多さ)、上位 15）
lw = torch.tensor([math.log(2 + wc[w]) for w in WORDS])
rev = {}
for view, val in ((0, "助詞:を"), (0, "助詞:が"), (0, "助詞:の"), (1, "助動詞:た"), (1, "助詞:を"), (1, "助動詞:ない"), (0, "名詞-普通名詞"), (1, "名詞-普通名詞")):
    if val in VI:
        s = pb[:, view, VI[val]] * lw
        rev[f"{['前', '後'][view]} = {val}"] = [WORDS[int(k)] for k in torch.argsort(s, descending=True)[:15]]
# 文の型: 品詞の値ごとの代表の語板（その値のよく出る語の平均）で、後の観点のいちばん明るい値を順にたどる
by_lab = collections.defaultdict(collections.Counter)
for s in SEQ["train"]:
    for w, l, _ in s:
        if w in WI:
            by_lab[l][w] += 1


def chain(start, n=7):
    out = [start]; cur = start
    for _ in range(n):
        ws = [WI[w] for w, _ in by_lab[cur].most_common(50)]
        if not ws:
            break
        nxt = pb[ws, 1, :NV].mean(0)
        for v in torch.argsort(nxt, descending=True).tolist():
            if VALS[v] != cur:
                cur = VALS[v]; break
        out.append(cur)
    return " → ".join(out)


res["chains"] = [chain(s) for s in ("名詞-普通名詞", "代名詞", "動詞-一般", "形容詞-一般", "副詞", "連体詞") if s in VI]
res["reverse"] = rev; res["sec"] = time.time() - t0
with open("results/exp82.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(res, clusters=[(n, pc, m) for n, pc, m in clusters]), ensure_ascii=False) + "\n")
L = ["# 前後の品詞（実験82）", "", "持ち主の問い（2026-10-31）: 逆引きで構文生成に使える仕組みか、前も後ろも品詞を想定できる仕組みか。作り方: experiments/exp82_pos_neighbors.py。", "",
     "## 1. 文章から品詞が自然に出るか", "",
     f"よく出る 5000 語を、すぐ前・すぐ後の語の数の似かただけで 40 の仲間に分けた。辞書の品詞（大分類）との一致（純度）: **{res['purity']:.3f}**（でたらめに分けると {res['purity_random']:.3f}）。", "",
     "| 数 | 多い品詞 | 語の例 |", "|---|---|---|"]
for n, pc, m in clusters:
    L.append(f"| {n} | {'・'.join(f'{p} {c}' for p, c in pc)} | {'・'.join(m)} |")
L += ["", "## 2. 前・後の品詞の観点を持つ語板（2 万語、縦 128 × 板 16）", "",
      f"- 次の語の品詞の当たり（確かめ用の材料、80 通り）: 語板 {res['board_bits_next']:.3f} ビット（上位 1 {res['board_top1_next']:.3f}）、表 {res['table_bits_next']:.3f}（{res['table_top1_next']:.3f}）。前の語の品詞: 語板 {res['board_bits_prev']:.3f}、表 {res['table_bits_prev']:.3f}。", "",
      "### 逆引き（その位置に来られる語）", ""] + [f"- **{k}**: {'・'.join(v)}" for k, v in rev.items()]
L += ["", "### 正引きの連鎖（文の型）", ""] + [f"- {c}" for c in res["chains"]]
open("docs/pos_neighbors_82.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
