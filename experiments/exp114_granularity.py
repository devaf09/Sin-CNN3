"""実験114: 区切りの粒（文字・短い語・長いかたまり）で、どれが一番「当たる」か（持ち主「区切りの粒は語板の長いほうが正解率が高いとかある？」）

公平な物差し: 1 文字あたり何ビットで、次を当てられるか（試しの文の 文字の量 で割る。粒が違っても同じ文を同じ文字の量で比べられる）。
  単位の 2 つ組（前の単位 → 次の単位）の数え上げで次の単位を当てる（Witten-Bell でならす）。
  教えに無い単位は、文字の 2 つ組でつづりを当てる分のビットを払う（長い粒ほど、知らない単位が増えて払いが増える）。
粒: 文字 1 つずつ / 学んだ区切り（実験113、区切りの数を 0.25〜1.5 倍で変える = 長いかたまり 〜 短い単位）/ 答えの区切り（英語は空白、日本語は Sudachi の短い単位）。
材料: Tatoeba の英語・日本語（CC BY 2.0 FR）。教え 10 万文、試し 2000 文（実験113 と同じ分け方）。
結果: results/exp114.txt、docs/reverse_use.md に追記、図 docs/figs/granularity_114.svg。実行: python experiments/exp114_granularity.py
"""
import json, math, random, sys
from collections import Counter, defaultdict

src = open("experiments/exp113_learn_segmentation.py", encoding="utf-8").read().split("def run(")[0]
exec(src)                                                                                      # 材料の読み込み（実験113）


class Seg:
    """実験113 の区切り（前後のばらけの山）。"""
    def __init__(self, train, N=4):
        self.N = N; self.fwd = [defaultdict(Counter) for _ in range(N + 1)]; self.bwd = [defaultdict(Counter) for _ in range(N + 1)]
        for ws in train:
            s = "^" + "".join(ws) + "$"
            for n in range(1, N + 1):
                for i in range(len(s) - n):
                    self.fwd[n][s[i:i + n]][s[i + n]] += 1; self.bwd[n][s[i + 1:i + 1 + n]][s[i]] += 1
        self.cache = {}
        self.gold_rate = sum(len(ws) - 1 for ws in train[:5000]) / sum(len("".join(ws)) - 1 for ws in train[:5000])
        self.allsc = sorted((x for ws in train[:3000] for x in self.scores("".join(ws))), reverse=True)

    def H(self, tab, n, key):
        k = (n, key, tab is self.fwd)
        if k not in self.cache:
            c = tab[n].get(key, {}); t = sum(c.values())
            self.cache[k] = -sum(v / t * math.log(v / t) for v in c.values()) if t else 0.0
        return self.cache[k]

    def scores(self, s):
        t = "^" + s + "$"; sc = []
        for i in range(1, len(s)):
            v = m = 0
            for n in range(1, self.N + 1):
                if i + 1 - n >= 0:
                    v += self.H(self.fwd, n, t[i + 1 - n:i + 1]); m += 1
                if i + 1 + n <= len(t):
                    v += self.H(self.bwd, n, t[i + 1:i + 1 + n]); m += 1
            sc.append(v / max(m, 1))
        return sc

    def cut(self, s, rate):
        th = self.allsc[int(len(self.allsc) * min(self.gold_rate * rate, 0.95))]; sc = self.scores(s)
        P = [i + 1 for i in range(len(sc)) if sc[i] >= th and (i == 0 or sc[i] >= sc[i - 1]) and (i == len(sc) - 1 or sc[i] >= sc[i + 1])]
        out, p = [], 0
        for q in P + [len(s)]:
            out.append(s[p:q]); p = q
        return out


def bits_per_char(train_units, test_units):
    """単位の 2 つ組（Witten-Bell）+ 知らない単位は文字の 2 つ組でつづる。1 文字あたりのビット。"""
    E = "</s>"
    c1 = Counter(); c2 = defaultdict(Counter)
    for us in train_units:
        prev = "<s>"
        for u in us + [E]:
            c1[u] += 1; c2[prev][u] += 1; prev = u
    N = sum(c1.values()); T1 = len(c1)
    ch1 = Counter(); ch2 = defaultdict(Counter)                                                 # つづり（文字の 2 つ組、単位の終わり付き）
    for u in c1:
        if u == E:
            continue
        prev = "<w>"
        for ch in list(u) + ["</w>"]:
            ch1[ch] += 1; ch2[prev][ch] += 1; prev = ch
    nch = len(ch1) + 50

    def spell(u):
        lp = 0.0; prev = "<w>"
        for ch in list(u) + ["</w>"]:
            c = ch2[prev]; t = sum(c.values()); d = len(c)
            p = ((c[ch] + d * ((ch1[ch] + 1) / (sum(ch1.values()) + nch))) / (t + d)) if t else (ch1[ch] + 1) / (sum(ch1.values()) + nch)
            lp += math.log2(p); prev = ch
        return lp

    def p1(u):
        return (c1[u] + T1 * (2 ** spell(u) if u != E else 1e-9)) / (N + T1)

    bits = 0.0; chars = 0
    for us in test_units:
        prev = "<s>"; chars += sum(len(u) for u in us)
        for u in us + [E]:
            c = c2[prev]; t = sum(c.values()); d = len(c)
            p = (c[u] + d * p1(u)) / (t + d) if t else p1(u)
            bits -= math.log2(p); prev = u
    return bits / chars


res = dict(exp="114", runs=[])
for lang, loader in (("英語", load_eng), ("日本語", load_jpn)):
    W = loader(); random.Random(0).shuffle(W); test = W[:2000]; train = W[2000:102000]
    seg = Seg(train)
    cfgs = [("文字 1 つずつ", lambda ws: list("".join(ws)))]
    for rate in (0.25, 0.5, 0.75, 1.0, 1.5):
        cfgs.append((f"学んだ区切り ×{rate}", lambda ws, r=rate: seg.cut("".join(ws), r)))
    cfgs.append(("答えの区切り（英語 = 空白、日本語 = 短い単位）", lambda ws: list(ws)))
    for name, f in cfgs:
        tru = [f(ws) for ws in train]; teu = [f(ws) for ws in test]
        bpc = bits_per_char(tru, teu); avg = sum(len("".join(u)) for u in teu) / max(sum(len(u) for u in teu), 1)
        vocab = len({u for us in tru for u in us})
        res["runs"].append(dict(lang=lang, name=name, bpc=bpc, chars_per_unit=avg, vocab=vocab))
        print(lang, name, "1 文字あたり", round(bpc, 3), "ビット", "単位の長さ", round(avg, 2), "単位の種類", vocab, flush=True)
with open("results/exp114.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
