"""実験113: 語の区切りを、文字の並びだけから学ぶ（実験112 で残った「言語ごとの部分」= 語の切り方、持ち主「古代語や宇宙人とも話せる AI」）

考え（言語の知識を入れない）: 語の中では次の文字がよく決まっていて、語の境目では次の文字がばらける（Harris の考え）。
  区切りの点（位置 i）= 前の n 文字のあとに来る文字のばらけ（エントロピー）+ 後ろの n 文字の前に来る文字のばらけ（n = 1〜4 の平均）。
  点が両隣より高く（山）、しきい値を超えた所で区切る。しきい値は、区切りの数が文字の数の何割かで決める（英語・日本語とも同じ決め方）。
  比べ: 区切らない、毎文字で区切る、でたらめに同じ数だけ区切る。
答え合わせだけに使うもの: 英語は元の空白（空白を消した文字の並びで学ぶ）、日本語は Sudachi の区切り（短い単位）。
材料: Tatoeba の英語・日本語（CC BY 2.0 FR）各 10 万文で数え、別の 2000 文で試す。評価 = 区切りの位置の 適合率・再現率・F1。
結果: results/exp113.txt、docs/reverse_use.md に追記、図 docs/figs/segmentation_113.svg。実行: python experiments/exp113_learn_segmentation.py
"""
import json, math, random, re, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"


def load_eng():
    out = []
    for l in open(f"{PRIV}/data/text/tatoeba_eng.jsonl", encoding="utf-8"):
        ws = re.findall(r"[a-z0-9']+", json.loads(l)["text"].lower())
        if len(ws) >= 2:
            out.append(ws)
    return out


def load_jpn():
    from sudachipy import Dictionary
    tok = Dictionary(dict="core").tokenizer(); out = []
    lines = [x for l in open(f"{PRIV}/data/text/tatoeba_jpn.jsonl", encoding="utf-8") for x in json.loads(l)["text"].split("\n") if x.strip()]
    random.Random(5).shuffle(lines)
    for x in lines[:110000]:
        ws = [m.surface() for m in tok.tokenize(x) if not m.part_of_speech()[0].startswith(("補助記号", "空白"))]
        if len(ws) >= 2:
            out.append(ws)
    return out


def bounds(ws):
    """語の列 → 文字の並びと、区切りの位置（i = i 文字目の前で区切る、1..len-1）。"""
    s = "".join(ws); B = set(); p = 0
    for w in ws[:-1]:
        p += len(w); B.add(p)
    return s, B


def run(lang, W, test_override=None):
    random.Random(0).shuffle(W); test = W[:2000] if test_override is None else test_override; train = W[2000:102000]
    N = 4; fwd = [defaultdict(Counter) for _ in range(N + 1)]; bwd = [defaultdict(Counter) for _ in range(N + 1)]
    for ws in train:
        s = "^" + "".join(ws) + "$"
        for n in range(1, N + 1):
            for i in range(len(s) - n):
                fwd[n][s[i:i + n]][s[i + n]] += 1                                               # n 文字のあとに来る文字
                bwd[n][s[i + 1:i + 1 + n]][s[i]] += 1                                           # n 文字の前に来る文字
    H = lambda c: (lambda t: -sum(v / t * math.log(v / t) for v in c.values()))(sum(c.values())) if c else 0.0
    cache = {}

    def ent(tab, n, key):
        k = (id(tab), n, key)
        if k not in cache:
            cache[k] = H(tab[n].get(key, {}))
        return cache[k]

    def scores(s):
        t = "^" + s + "$"; sc = []
        for i in range(1, len(s)):                                                              # s の i 文字目の前（t では i+1）
            v = 0.0; m = 0
            for n in range(1, N + 1):
                if i + 1 - n >= 0:
                    v += ent(fwd, n, t[i + 1 - n:i + 1]); m += 1                                # 左の n 文字のあとのばらけ
                if i + 1 + n <= len(t):
                    v += ent(bwd, n, t[i + 1:i + 1 + n]); m += 1                                # 右の n 文字の前のばらけ
            sc.append(v / max(m, 1))
        return sc

    gold_rate = sum(len(bounds(ws)[1]) for ws in train[:5000]) / sum(len("".join(ws)) - 1 for ws in train[:5000])
    allsc = [x for ws in train[:3000] for x in scores("".join(ws))]
    out = []
    for rate in (0.5, 1.0, 1.5):                                                                # 区切りの数を「答えの区切りの割合」の何倍にするか（しきい値の決め方）
        th = sorted(allsc, reverse=True)[int(len(allsc) * min(gold_rate * rate, 0.95))]
        tp = fp = fn = 0
        for ws in test:
            s, G = bounds(ws); sc = scores(s)
            P = {i + 1 for i in range(len(sc)) if sc[i] >= th and (i == 0 or sc[i] >= sc[i - 1]) and (i == len(sc) - 1 or sc[i] >= sc[i + 1])}
            tp += len(P & G); fp += len(P - G); fn += len(G - P)
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        out.append(dict(lang=lang, rate=rate, prec=p, rec=r, f1=2 * p * r / max(p + r, 1e-9)))
        print(lang, "区切りの多さ ×", rate, "適合", round(p, 3), "再現", round(r, 3), "F1", round(out[-1]["f1"], 3), flush=True)
    # 比べ: でたらめ（答えの数だけ）・毎文字
    tp = fp = fn = 0; tpa = fpa = fna = 0; rng = random.Random(1)
    for ws in test:
        s, G = bounds(ws); k = len(G); P = set(rng.sample(range(1, len(s)), k)) if len(s) > 1 else set()
        tp += len(P & G); fp += len(P - G); fn += len(G - P)
        A = set(range(1, len(s))); tpa += len(A & G); fpa += len(A - G); fna += len(G - A)
    f = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)
    out.append(dict(lang=lang, rate="でたらめ", f1=f(tp, fp, fn))); out.append(dict(lang=lang, rate="毎文字", f1=f(tpa, fpa, fna)))
    print(lang, "でたらめ F1", round(f(tp, fp, fn), 3), "毎文字 F1", round(f(tpa, fpa, fna), 3), flush=True)
    # 例
    ex = []
    best = max((o for o in out if isinstance(o["rate"], float)), key=lambda o: o["f1"])
    th = sorted(allsc, reverse=True)[int(len(allsc) * min(gold_rate * best["rate"], 0.95))]
    for ws in test[:6]:
        s, G = bounds(ws); sc = scores(s)
        P = {i + 1 for i in range(len(sc)) if sc[i] >= th and (i == 0 or sc[i] >= sc[i - 1]) and (i == len(sc) - 1 or sc[i] >= sc[i + 1])}
        ex.append(dict(answer=" ".join(ws), learned="".join((" " if i in P else "") + ch for i, ch in enumerate(s))))
    return out, ex


def load_bunsetsu_test():
    """日本語の答え合わせを、短い単位（Sudachi）ではなく文節（GiNZA、実験107 の試しの文）の区切りでも見る。記号は外す。"""
    S = json.load(open(f"{PRIV}/data/views/sentence_trees.json", encoding="utf-8")); out = []
    for x in S:
        if x["split"] != "test":
            continue
        ws = ["".join(w[0] for w in b["w"] if not w[2].startswith(("補助記号", "空白"))) for b in x["b"]]
        ws = [w for w in ws if w]
        if len(ws) >= 2:
            out.append(ws)
    return out[:2000]


res = dict(exp="113", runs=[], examples={})
JP = load_jpn()
for lang, W, tst in (("英語", load_eng(), None), ("日本語（短い単位）", JP, None), ("日本語（文節）", JP, load_bunsetsu_test())):
    o, e = run(lang, W, tst); res["runs"] += o
    if tst is None:
        res["examples"][lang] = e                                                              # 文節の試しは会話コーパスの文なので、例は残さない
    for x in e:
        print(" ", x, flush=True)
with open("results/exp113.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
