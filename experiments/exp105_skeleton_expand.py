"""実験105: 骨格生成 → 系列展開 の 2 段目（持ち主が見つけた案 2026-11-02「骨格を作ってから、各スロットを局所的に並列展開する」）

骨格のスロット = 抽象的な項目だけ: 役割（GiNZA の係り方 nsubj・obj・obl・advcl…）、中身の語の基本形（太郎・見付ける）、その品詞、
                述語なら 時制・否定・丁寧 の印（過去 = た、否定 = ない・ぬ・ず・ん、丁寧 = ます・です。骨格が決める意味の一部）。
展開 = スロットから、具体的な文節（中身の語の活用した形 + 助詞・助動詞・句読点の尾）を作る。どのスロットも独立に、並列に作れる。
  尾の出し方は 2 通り:
    まとめて: 尾全体（例「に」「られ て いる」「まし た 。」）を 1 つの組として、数え上げの表で選ぶ。
    1 語ずつ（局所的な因果の畳み込み）: 尾の語を、スロットの印 + 直前の 2 語 から 1 語ずつ選ぶ（そのスロットの中だけを見る）。
  中身の語の形: (基本形, 尾の最初の語) で、学習で一番多かった形（いじめ + られ、見つけ + ます）。
参照光（骨格全体から配る光）: 係る先の文節の中身の語（その述語）。「亀 + 乗る → 亀に」「亀 + 見る → 亀を」を決めるのに要るはず。
比べ: 0 = 役割・品詞・印だけ、1 = + 中身の語、2 = + 参照光（係る先の語）、3 = 1 + 2（後ろから順に、数え上げが無ければ一つ前へ戻る）。
材料: Sin-CNN3_private/data/views/bunsetsu_roles.json（3 万文、文節 22 万。教え 19.8 万・試し 2.2 万。実験だけに使う材料を含むので、例は浦島太郎だけを載せる）。
試し: 試しの文節 + 浦島太郎の 20 文（80 枠の文節）。正解 = 文節の文字がそのまま一致。尾だけの一致も見る。
結果: results/exp105.txt、docs/skeleton_expand_105.md、図 docs/figs/skeleton_expand_105.svg。実行: python experiments/exp105_skeleton_expand.py
"""
import json, os, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
PRIV = "D:/AI_git/Sin-CNN3_private"
FUNC = ("助詞", "助動詞", "補助記号")                                                        # 接尾辞（さん・たち・日）は中身に入れる
NEG = {"ない", "ぬ", "ず", "ん", "無い"}


def split_bunsetsu(words):
    """文節の語 [(文字, 基本形, 品詞)] を、中身（最初の続き）と尾（助詞・助動詞・非自立の語・句読点）に分ける。"""
    i = 0
    while i < len(words) and not words[i][2].startswith(FUNC) and ("非自立" not in words[i][2] or i == 0):
        i += 1
    i = max(i, 1)
    return words[:i], words[i:]


def slot_of(words, dep, head_words):
    """骨格のスロット（抽象的な項目）と、正解の文節（中身の形・尾）。"""
    core, tail = split_bunsetsu(words)
    lemma = "".join(w[1] for w in core); pos = core[-1][2].split("-")[0]
    tl = [w[1] for w in tail]
    flags = ""
    if pos in ("動詞", "形容詞") or dep == "ROOT":                                               # 述語の意味の印（骨格が決める）: 時制・否定・丁寧・受け身・続き・望み
        flags = "".join(f for f, hit in (("過去", "た" in tl), ("否定", bool(NEG & set(tl))), ("丁寧", bool({"ます", "です"} & set(tl))),
                                          ("受け身", bool({"れる", "られる"} & set(tl))), ("続き", "て" in tl and bool({"居る", "いる"} & set(tl))), ("望み", "たい" in tl)) if hit)
    case = next((w[1] for w in tail if w[2].startswith("助詞-格助詞") or w[2].startswith("助詞-係助詞")), "")
    head = split_bunsetsu(head_words)[0] if head_words else None
    ref = "".join(w[1] for w in head) if head else "（根）"
    return dict(dep=dep, lemma=lemma, pos=pos, flags=flags, ref=ref, case=dep + ":" + case), "".join(w[0] for w in core), tuple(w[0] for w in tail)


rows = json.load(open(f"{PRIV}/data/views/bunsetsu_roles.json", encoding="utf-8"))
data = {"train": [], "test": []}
for r in rows:
    w = [(x[0], x[3], x[2]) for x in r["words"]]; hw = [(x[0], x[3], x[2]) for x in r["head_words"]] if r["head_words"] and not r["is_root"] else None
    data[r["split"]].append(slot_of(w, "ROOT" if r["is_root"] else r["dep"], hw))
U = []                                                                                          # 浦島太郎の文節
for s in json.load(open("data/urashima_parse.json", encoding="utf-8")):
    B = s["bunsetsu"]
    for b in B:
        w = [tuple(x[:3]) for x in b["words"]]; hw = [tuple(x[:3]) for x in B[b["head"]]["words"]] if b["head"] != b["id"] and b["dep"] != "ROOT" else None
        U.append(slot_of(w, b["dep"], hw) + (s["sentence"], b["text"]))
print("教え", len(data["train"]), "試し", len(data["test"]), "浦島太郎", len(U), flush=True)
KEYS = {0: ("dep", "pos", "flags"), 1: ("dep", "pos", "flags", "lemma"), 2: ("dep", "pos", "flags", "ref"), 3: ("dep", "pos", "flags", "lemma", "ref"),
        4: ("case", "pos", "flags"), 5: ("case", "pos", "flags", "lemma", "ref")}   # 4・5: 骨格が「格（が・は・に・で…）」まで決める場合
CHAIN = {0: [0], 1: [1, 0], 2: [2, 0], 3: [3, 2, 1, 0], 4: [4], 5: [5, 4]}                                        # 数え上げが無いときに戻る順
key = lambda s, m: tuple(s[k] for k in KEYS[m])
# ---- 数え上げ（学習） ----
whole = {m: defaultdict(Counter) for m in KEYS}                                                 # まとめて: スロット → 尾
step = {m: defaultdict(Counter) for m in KEYS}                                                  # 1 語ずつ: (スロット, 直前 2 語) → 次の語（「終」で止まる）
form = defaultdict(Counter); form1 = defaultdict(Counter)                                       # 中身の形
for s, core, tail in data["train"]:
    for m in KEYS:
        whole[m][key(s, m)][tail] += 1
        ctx = ("始", "始")
        for t in tail + ("終",):
            step[m][key(s, m) + ctx][t] += 1; ctx = (ctx[1], t)
    form[(s["lemma"], tail[0] if tail else "終")][core] += 1; form1[s["lemma"]][core] += 1


MINC = 3


def pick(table, s, m, extra=()):
    for mm in CHAIN[m]:
        c = table[mm].get(key(s, mm) + extra)
        if c and (sum(c.values()) >= MINC or mm == CHAIN[m][-1]):                               # 数が少ない（MINC 未満の）表は信じずに一つ前へ戻る
            return c.most_common(1)[0][0]
    return None


def expand(s, m, how):
    if how == "まとめて":
        tail = pick(whole, s, m) or ()
    else:
        tail, ctx = [], ("始", "始")
        for _ in range(6):                                                                      # 尾は長くても 6 語
            t = pick(step, s, m, ctx)
            if t is None or t == "終":
                break
            tail.append(t); ctx = (ctx[1], t)
        tail = tuple(tail)
    c = form.get((s["lemma"], tail[0] if tail else "終")) or form1.get(s["lemma"])
    core = c.most_common(1)[0][0] if c else s["lemma"]
    return core, tail


res = dict(exp="105", n_train=len(data["train"]), n_test=len(data["test"]), runs=[])
EX = {}
for how in ("まとめて", "1 語ずつ"):
    for m in KEYS:
        out = {}
        for name, D in (("試し", data["test"]), ("浦島太郎", U)):
            ok = tok = tnp = 0; by = Counter(); byn = Counter()
            for item in D:
                s, core, tail = item[:3]
                c2, t2 = expand(s, m, how); good = (c2 + "".join(t2)) == (core + "".join(tail))
                ok += good; tok += t2 == tail; nop = lambda t: tuple(x for x in t if x not in ("、", "。", "，", "．")); tnp += nop(t2) == nop(tail); by[s["dep"]] += good; byn[s["dep"]] += 1
                if name == "浦島太郎":
                    EX.setdefault((how, m), []).append((item[3], item[4], s, c2 + "".join(t2), good))
            out[name] = dict(n=len(D), whole=ok / len(D), tail=tok / len(D), tail_nopunct=tnp / len(D), by_dep={d: round(by[d] / byn[d], 3) for d, _ in byn.most_common(8)})
        res["runs"].append(dict(how=how, model=m, **out))
        print(how, m, "試し 文節", round(out["試し"]["whole"], 3), "尾", round(out["試し"]["tail"], 3), "句読点なし", round(out["試し"]["tail_nopunct"], 3), "浦島 文節", round(out["浦島太郎"]["whole"], 3), "尾", round(out["浦島太郎"]["tail"], 3), "句読点なし", round(out["浦島太郎"]["tail_nopunct"], 3), flush=True)
with open("results/exp105.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
# 浦島太郎: 骨格（スロットだけ）→ 展開した文（1 語ずつ・モデル 3 と、0 の比べ）
L = []
for (how, m) in (("1 語ずつ", 3), ("1 語ずつ", 0), ("1 語ずつ", 5)):
    L.append(f"## 浦島太郎の展開（{how}・比べ {m}）\n")
    cur = None; got = []; want = []
    for sent, text, s, outp, good in EX[(how, m)] + [(None, None, None, None, None)]:
        if sent != cur and cur is not None:
            L.append(f"- 元: {cur}\n  - 展開: {''.join(got)}（{sum(want)}/{len(want)} 文節）")
            got, want = [], []
        if sent is None:
            break
        cur = sent; got.append(outp if good else f"[{outp}]"); want.append(bool(good))
    L.append("")
json.dump({f"{h}|{m}": [(t, o, g, s) for _, t, s, o, g in v] for (h, m), v in EX.items()}, open("results/exp105_urashima.json", "w", encoding="utf-8"), ensure_ascii=False)
open("results/exp105_urashima.md", "w", encoding="utf-8").write("\n".join(L))
