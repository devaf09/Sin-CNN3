"""実験71 の例: 有名な小説 10 冊で、人物などの語を含む行（連想元）から、別の章の関係の深い行（連想先）を、木の道筋の目印で出す。
出力: docs/examples_71.md。実行: python tools/exp71_examples.py"""

import collections, json, math, os, random, re, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t00 = time.time()
M = torch.load("D:/AI_git/Sin-CNN3_private/models/goban_名詞_seed0.pt", weights_only=False)
WORDS = M["words"]; WI = {w: i for i, w in enumerate(WORDS)}
with torch.no_grad():
    B3 = torch.cat([run(M["P"][i:i + 5000], 5, 52) for i in range(0, len(WORDS), 5000)])
B3 = B3 / B3.sum(-1, keepdim=True).clamp(min=1e-12)                                         # (語, 観点, 値) の割合
B = B3.reshape(len(WORDS), -1)                                                              # 1 語 = 260 個の明るさの模様
# ---- 属性樹（実験65 の考え、2 万語） ----
ATTR = [(v, x) for v in range(5) for x in range(47)]                                        # 属性 = (観点, 値)（その他は除く）
prior = B3.mean(0)
has = torch.stack([B3[:, v, x] >= max(0.05, 3 * float(prior[v, x])) for v, x in ATTR], 1)    # (語, 属性) 持つか
nodes = [dict(depth=0, members=torch.arange(len(WORDS)), path=[])]                         # 根 = 全部の語


def grow(i, used):
    nd = nodes[i]
    if nd["depth"] >= 4 or len(nd["members"]) < 20:
        return
    share = has[nd["members"]].float().mean(0)
    for a in used:
        share[a] = -1
    for a in torch.argsort(share, descending=True)[:6].tolist():                            # 共通の度合いの大きい属性 6 つで枝
        if share[a] < 0.05:
            break
        mem = nd["members"][has[nd["members"], a]]
        if len(mem) < 10 or len(mem) == len(nd["members"]):
            continue
        nodes.append(dict(depth=nd["depth"] + 1, members=mem, path=nd["path"] + [a])); grow(len(nodes) - 1, used + [a])


grow(0, [])
fused = torch.stack([B[nd["members"]].mean(0) for nd in nodes])                            # 節の融合した板
deep = torch.zeros(len(WORDS), dtype=torch.long); deep_d = torch.zeros(len(WORDS), dtype=torch.long)
tag = torch.zeros(len(WORDS), len(nodes))                                                   # 語が通る節の印（深いほど重く）
for j, nd in enumerate(nodes):
    m = nd["members"]; tag[m, j] = 2.0 ** nd["depth"]
    better = nd["depth"] > deep_d[m]; deep[m[better]] = j; deep_d[m[better]] = nd["depth"]
print("属性樹の節", len(nodes), "深さの分布", collections.Counter(deep_d.tolist()), f"{time.time() - t00:.0f} 秒", flush=True)
smooth = {a: (1 - a) * B + a * fused[deep] for a in (0.3, 0.5)}                              # 木でならした語板

# ---- 有名な小説で、連想元と連想先の行を例として出す ----
NV = M["vals"]; VIEWS = list(NV.keys())
def attr_name(a):
    v, x = ATTR[a]; r = (x - 9 * v) % 47
    return f"{VIEWS[v]} {NV[VIEWS[v]][r]}" if r < len(NV[VIEWS[v]]) else f"{VIEWS[v]} ?"
node_name = lambda j: " → ".join(attr_name(a) for a in nodes[j]["path"]) or "根"
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").create(); MODE = SplitMode.C
idx = json.load(open("D:/AI_git/Sin-CNN3_private/data/text/famous_index.json", encoding="utf-8"))
texts = {}
for l in open("D:/AI_git/Sin-CNN3_private/data/text/aozora_full.jsonl", encoding="utf-8"):
    d = json.loads(l); texts[d["id"]] = d["text"]
BOOKS = ["吾輩は猫である", "坊っちゃん", "こころ", "三四郎", "草枕", "それから", "人間失格", "斜陽", "銀河鉄道の夜", "風の又三郎"]
PEOPLE = "主人 細君 先生 奥さん 下女 婆さん 少年 男 女 母 父 兄 妹 娘 子供 友達 校長 教頭 生徒 叔父 妻 夫 姉 弟 女中 客 子 猫 先生 学生 母親 父親 老人 青年 友人 坊ちゃん 女学生 子ども 同級生".split()
out = ["# 章をまたぐ行の連想の例（実験71 の「木の道筋の目印」）", "",
       "青空文庫（著作権切れ）の有名な小説 10 冊で、人物などの語を 2〜3 個選び、その語を含む行（連想元）から、別の章でいちばん関係が深いと連想した行（連想先）を出した。",
       "連想元の語そのものは、両方の行の手がかりから取り除いている（同じ語を探すだけにならないように）。「つないだ節」は、両方の行の名詞が一緒に通った属性樹の節のうち、深いもの（理由）。",
       "行は長いものを 70 文字で切っている。作り方: tools/exp71_examples.py", ""]
for title in BOOKS:
    if title not in idx:
        continue
    zid, author = idx[title]; text = texts.get(zid)
    if not text:
        continue
    sents = [s for s in re.split(r"(?<=。)", text.replace(chr(10), "").replace("　", "")) if len(s) > 8 and "---" not in s and "テキスト中" not in s and "底本" not in s]   # 作品の頭・終わりの説明は除く
    lines = []
    for k, s in enumerate(sents):
        ns = sorted({m.normalized_form() for m in tok.tokenize(s[:500], MODE) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "普通名詞" and m.normalized_form() in WI})
        if len(ns) >= 2:
            lines.append((k * 4 // len(sents), ns, s))
    cnt = collections.Counter(n for _, ns, _ in lines for n in ns)
    words = [w for w in sorted(set(PEOPLE), key=lambda w: -cnt[w]) if cnt[w] >= 4][:3]
    out += [f"## {title}（{author}）", "", "| 語 | 連想元の行（章） | 連想先の行（章） | つないだ節（深い順） |", "|---|---|---|---|"]
    used = set()
    for w in words:
        qs = [li for li, (ch, ns, s) in enumerate(lines) if w in ns and len(ns) >= 4 and li not in used]   # 語ごとに別の行を連想元にする
        if not qs:
            continue
        a = max(qs, key=lambda li: len(lines[li][1])); used.add(a)
        feat = lambda li: tag[[WI[x] for x in lines[li][1] if x != w]].sum(0) if [x for x in lines[li][1] if x != w] else None
        qa = feat(a); ids = [li for li, (ch, ns, s) in enumerate(lines) if ch != lines[a][0] and feat(li) is not None]
        S = torch.nn.functional.cosine_similarity(torch.stack([feat(li) for li in ids]), qa[None], dim=1); b = ids[int(S.argmax())]
        shared = (tag[[WI[x] for x in lines[a][1] if x != w]].amax(0) > 0) & (tag[[WI[x] for x in lines[b][1] if x != w]].amax(0) > 0)
        js = sorted([j for j in torch.nonzero(shared).flatten().tolist() if nodes[j]["depth"] > 0], key=lambda j: -nodes[j]["depth"])[:2]
        cut = lambda s: (s[:70] + "…") if len(s) > 70 else s
        out.append(f"| {w} | {cut(lines[a][2])}（{lines[a][0] + 1}） | {cut(lines[b][2])}（{lines[b][0] + 1}）{'（同じ語を含む）' if w in lines[b][1] else ''} | {'<br>'.join(node_name(j) for j in js) or '−'} |")
    out.append("")
open("docs/examples_71.md", "w", encoding="utf-8").write(chr(10).join(out) + chr(10))
print(chr(10).join(out[:40]))
