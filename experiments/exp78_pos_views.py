"""実験78: 品詞の観点（D 案）— 品詞を観点周波数の木にする。最初の枝 = 品詞の大分類、次の周波数から = 品詞ごとの中分類

着想: 2026-10-31 持ち主「品詞を観点周波数の最初の枝（名詞、動詞、形容詞…）で 1 つにし、次の周波数には動詞、次の周波数には形容詞、とできないか。
      複数の品詞（揺れ）を最初の枝で確率的に残して、確率的に関係する品詞をザクっと逆引きする仕組みが必要」。docs/grammar.md の 1 節。
方法:
  観点（周波数 1 から）:
    1 品詞（大分類 14）: 名詞・代名詞・動詞・形容詞・形状詞・副詞・連体詞・接続詞・感動詞・助動詞・助詞・接頭辞・接尾辞・記号
    2 名詞（普通名詞・固有名詞-地名・固有名詞-人名・固有名詞-組織・固有名詞-一般・数詞・助動詞語幹）
    3 動詞（一般・非自立可能） 4 形容詞（一般・非自立可能） 5 形状詞（一般・タリ・助動詞語幹） 6 感動詞（一般・フィラー）
    7 接尾辞（名詞的・形容詞的・形状詞的・動詞的） 8 助詞（格・係・副・接続・終・準体）
  語 = 材料で学習用の側に 5 回以上出た語（正規形）のうち多い順 5 万語 ＋ 上限のある品詞の語は全部。値 = その語が材料の中で実際になった品詞の割合（揺れ）。
  その品詞にならない語は、その品詞の観点を使わない（学習しない印）。器 = 縦 64 × 板 16（値が少ないので縦 64 で足りる）、500 回。
  確かめ: (1) 確かめ用の材料の品詞の当たり（交差エントロピー、表との差）、(2) 揺れのある語（が・で・と・から・の…）の割合、
          (3) 観点ごとの逆引き（最初の枝「接続詞」「接頭辞」「感動詞」などを逆に入れて、反応する語 = 確率の重みつき）。
材料: 会話（RealPersonaChat・JMultiWOZ）全部・青空文庫 1000 作品・Web 小説 1 万話（実験だけ、数だけ）。学習用と確かめ用は文書ごとに 9:1。
結果: results/exp78.txt、docs/pos_views_78.md。語板: 非公開フォルダの models/goban_pos.pt
実行: python experiments/exp78_pos_views.py --device cuda（experiments/sweeps/sweep78.sh）
"""
import argparse, collections, json, os, random, re, sys, time, zlib

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--words", type=int, default=50000)
a = ap.parse_args()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob, evaluate

sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time(); dev = a.device; PRIV = "D:/AI_git/Sin-CNN3_private"
# ---- 品詞の木（観点と値） ----
TOP = ["名詞", "代名詞", "動詞", "形容詞", "形状詞", "副詞", "連体詞", "接続詞", "感動詞", "助動詞", "助詞", "接頭辞", "接尾辞", "記号"]
SUB = {"名詞": ["普通名詞", "固有名詞-地名", "固有名詞-人名", "固有名詞-組織", "固有名詞-一般", "数詞", "助動詞語幹"],
       "動詞": ["一般", "非自立可能"], "形容詞": ["一般", "非自立可能"], "形状詞": ["一般", "タリ", "助動詞語幹"], "感動詞": ["一般", "フィラー"],
       "接尾辞": ["名詞的", "形容詞的", "形状詞的", "動詞的"], "助詞": ["格助詞", "係助詞", "副助詞", "接続助詞", "終助詞", "準体助詞"]}
VIEWS = ["品詞"] + list(SUB)                                                                 # 観点 1 = 品詞の大分類、2〜8 = 品詞ごとの中分類
NV = max(len(TOP), max(len(v) for v in SUB.values()))                                       # 1 つの観点の値の数の最大（14）
CLOSED = {"助詞", "助動詞", "接続詞", "連体詞", "代名詞"}


def key(p):
    """Sudachi の品詞（6 段）→ (大分類, 中分類)。記号・補助記号・空白は「記号」にまとめる。固有名詞は地名・人名・組織・一般に分ける。"""
    big = "記号" if p[0] in ("記号", "補助記号", "空白") else p[0]
    if big == "名詞" and p[1] == "固有名詞":
        sub = "固有名詞-" + ("地名" if p[2] == "地名" else "人名" if p[2] == "人名" else "組織" if p[2] == "組織" else "一般")
    else:
        sub = p[1]
    return big, sub


# ---- 材料で、語ごとの品詞を数える（学習用・確かめ用） ----
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").tokenizer(); M = SplitMode.C
C = {s: collections.defaultdict(collections.Counter) for s in ("train", "test")}


def count(text, did):
    s = "test" if zlib.crc32(did.encode()) % 10 == 0 else "train"
    for sent in re.split(r"(?<=[。！？\n])", text):
        sent = sent.strip()
        if sent and len(sent) <= 500:
            for m in tok.tokenize(sent, M):
                C[s][m.normalized_form()][key(m.part_of_speech())] += 1


QUICK = os.environ.get("QUICK") == "1"                                                      # 試しに動かすとき: 材料を少なく
for n_l, l in enumerate(open(f"{PRIV}/data/text/conv_all.jsonl", encoding="utf-8")):
    if QUICK and n_l > 2000:
        break
    d = json.loads(l)
    if not d["id"].startswith("aozora"):
        count(d["text"], "conv:" + d["id"])
A = [json.loads(l) for l in open(f"{PRIV}/data/text/aozora_full.jsonl", encoding="utf-8")]; random.Random(0).shuffle(A)
for d in A[:20 if QUICK else 1000]:
    count(d["text"], "aozora:" + d["id"])
del A
W = [json.loads(l) for l in open(f"{PRIV}/data/text/webnovels2.jsonl", encoding="utf-8")]; random.Random(0).shuffle(W)
for d in W[:100 if QUICK else 10000]:
    count(d["text"], "web:" + d["work"])                                                       # Web 小説は作品ごとに分ける
del W
print("数えた語", len(C["train"]), f"{time.time() - t0:.0f} 秒", flush=True)
tot = {w: sum(c.values()) for w, c in C["train"].items()}
closed = [w for w, c in C["train"].items() if any(b in CLOSED for b, _ in c)]                 # 上限のある品詞になりうる語は全部
WORDS = list(dict.fromkeys(closed + [w for w in sorted(tot, key=tot.get, reverse=True) if tot[w] >= 5][:a.words]))
WI = {w: i for i, w in enumerate(WORDS)}


def table(split):
    """(語, 観点, 値 + その他) の数。観点 1 = 大分類、その品詞の観点 = 中分類。"""
    T = torch.zeros(len(WORDS), len(VIEWS), NV + 1)
    for i, w in enumerate(WORDS):
        for (big, sub), m in C[split].get(w, {}).items():
            T[i, 0, TOP.index(big)] += m
            if big in SUB:
                j = VIEWS.index(big); T[i, j, SUB[big].index(sub) if sub in SUB[big] else NV] += m
    return T


TR, TE = table("train"), table("test")
use = TR.sum(2) > 0                                                                         # その品詞にならない語は、その観点を学ばない
pri = (TR.sum(0) + 0.5) / (TR.sum(0) + 0.5).sum(1, keepdim=True)
tg = (TR + 2 * pri[None]) / (TR.sum(2, keepdim=True) + 2)                                   # 少しだけ平らに（品詞はほぼ決まっているので k = 2）
print("語", len(WORDS), "（上限のある品詞になりうる語", len(closed), "）", f"{time.time() - t0:.0f} 秒", flush=True)
B = 5000
P = torch.cat([train(tg[i:i + B], use[i:i + B], a.steps, i, lr=0.03, dev=dev, rows=64, plates=16).cpu() for i in range(0, len(WORDS), B)])
Wt = torch.cat([run(P[i:i + B].to(dev), len(VIEWS), NV + 1).cpu() for i in range(0, len(WORDS), B)])
pb = prob(Wt)
mask = use[..., None].float()
res = dict(exp="78", words=len(WORDS), closed=len(closed), views=VIEWS, steps=a.steps,
           board=evaluate(pb * mask + (1 - mask) / (NV + 1), TE * mask)["bits"], table=evaluate(tg * mask + (1 - mask) / (NV + 1), TE * mask)["bits"],
           board_top=evaluate(pb[:, :1], TE[:, :1]), table_top=evaluate(tg[:, :1], TE[:, :1]))
# 揺れのある語の、最初の枝（大分類）の割合: 数えた割合と語板の割合
AMB = ["が", "で", "と", "から", "の", "に", "な", "ね", "や", "けれど", "そして", "又", "此の", "何", "大きな", "同じ", "まで", "ので", "ない", "らしい"]
amb = []
for w in AMB:
    if w in WI:
        i = WI[w]; t = TR[i, 0] / TR[i, 0].sum().clamp(min=1); b = pb[i, 0]
        top = torch.argsort(t[:len(TOP)], descending=True)[:3]
        amb.append((w, [(TOP[int(k)], round(float(t[k]), 3), round(float(b[k]), 3)) for k in top if t[k] > 0.005]))
# 観点ごとの逆引き: 最初の枝の値を逆に入れる → 各語板の入口に戻る光（= その値の明るさ）× 語の多さ（重み）。上位 15 語
rev = {}
logf = torch.tensor([float(torch.log(torch.tensor(2.0 + tot.get(w, 0)))) for w in WORDS])
for t in ["接続詞", "接頭辞", "感動詞", "連体詞", "代名詞", "形状詞"]:
    s = pb[:, 0, TOP.index(t)] * logf
    rev[t] = [(WORDS[int(k)], round(float(pb[int(k), 0, TOP.index(t)]), 2)) for k in torch.argsort(s, descending=True)[:15]]
res.update(ambiguous=amb, reverse=rev, sec=time.time() - t0, device=dev)
torch.save(dict(P=P, words=WORDS, views=VIEWS, top=TOP, sub=SUB, table=tg), f"{PRIV}/models/goban_pos.pt")
with open("results/exp78.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
L = ["# 品詞の観点（実験78、D 案）", "", "品詞を観点周波数の木にした（観点 1 = 大分類、観点 2〜8 = 品詞ごとの中分類）。作り方: experiments/exp78_pos_views.py、計画: [文法解析・組立](grammar.md)。", "",
     f"- 語板 {len(WORDS):,} 語（上限のある品詞になりうる語 {len(closed):,} を含む）。器 = 縦 64 × 板 16、{a.steps} 回。",
     f"- 確かめ用の材料での当たり（使う観点だけ、ビット）: 語板 {res['board']:.3f}、数えた表 {res['table']:.3f}。最初の枝（大分類）だけ: 語板 {res['board_top']['bits']:.3f}（上位 1 の当たり {res['board_top']['top1']:.3f}）、表 {res['table_top']['bits']:.3f}。", "",
     "## 揺れのある語（最初の枝の割合: 数えた割合 → 語板の割合）", "", "| 語 | 品詞の割合 |", "|---|---|"]
for w, xs in amb:
    L.append(f"| {w} | " + "、".join(f"{t} {x:.3f} → {y:.3f}" for t, x, y in xs) + " |")
L += ["", "## 最初の枝からのザクっと逆引き（その品詞の明るさ × 語の多さ、上位 15）", ""]
for t, xs in rev.items():
    L.append(f"- **{t}**: " + "・".join(f"{w}（{p}）" for w, p in xs))
open("docs/pos_views_78.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
