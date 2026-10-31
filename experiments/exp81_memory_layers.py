"""実験81-3: 短期・中期の記憶（持ち主の第 3 案 3-1）— 名札の数に上限を置いたとき、何を残すと良いか

着想: 持ち主「短期・中期記憶グレイン群を作り、小さな語板群のように属性と名札のセットを持つ。その間だけ有効で、薄まっていった名札は捨てられて
      新しい名札と入れ替えられる。ずっと残る記憶は最終的に長期記憶（既存の語板）を更新する」「何を残すかは自覚がいいと思うけど、実験で試すしかない」。
問題: 実験81 と同じ当て方（各文の名詞を前の文だけから、3 万語の中の順位）。ただし覚えておく長さが効くように、同じ作品の続いた 10 話を 1 つの会話（読み）とする。
      確かめ用 14 作品から 40 か所。名札の記憶 = 実験81-1 の混ぜ合わせ（λ = 0.7）。名札の数の上限 C = 20・50・200・上限なし。
残す基準（上限を超えたら、点の低い名札を捨てる）:
  新しさ: いちばん最近使われた名札を残す（最近使われていないものを捨てる）
  回数: 出た回数の多い名札を残す
  光の強さ: 出るたびに 1 を足し、1 文ごとに 0.97 倍に薄まる強さ（持ち主の「薄まっていった名札は捨てる」）
  自覚（意外さ）: 初めて出たときの長期の答えでの意外さ（−log P(語 | 長期)、知らない・思いがけない語ほど高い）× 光の強さ
  2 段（短期 + 中期）: 短期 = 光の強さで上限 C/4、短期から捨てられた名札のうち 2 回以上出たものを中期（光の強さ、薄まり 0.995、上限 3C/4）へ
測るもの: MRR（全部・くり返し）。会話の終わりに中期に残った名札の数（= 長期の語板へ移す候補）。
測り方の限界: 語が出るかだけ。意味・否定は見ない。
結果: results/exp81.txt。実行: python experiments/exp81_memory_layers.py
"""
import collections, json, math, os, pickle, random, re, sys, time, zlib

import torch

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t0 = time.time()
PRIV = "D:/AI_git/Sin-CNN3_private"
NX = torch.load(f"{PRIV}/models/goban_next_web2.pt", weights_only=False)
WORDS = NX["words"]; WI = {w: i for i, w in enumerate(WORDS)}; LAB = NX["cluster"]; LOGB = NX["board"].clamp(min=1e-9).log()
FREQ = pickle.load(open(f"{PRIV}/data/views/merged2.pkl", "rb"))["noun_total"]
lpw = torch.tensor([math.log(1 + FREQ.get(w, 0)) for w in WORDS])
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").tokenizer(); MODE = SplitMode.C
nouns = lambda s: [m.normalized_form() for m in tok.tokenize(s[:500], MODE) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("普通名詞", "固有名詞") and m.normalized_form() in WI]
eps = collections.defaultdict(list)
for d in map(json.loads, open(f"{PRIV}/data/text/webnovels2.jsonl", encoding="utf-8")):
    if zlib.crc32(d["work"].encode()) % 10 == 0:
        eps[d["work"]].append(d)
sessions = []
for w, L in eps.items():
    L.sort(key=lambda d: d["ep"])
    for s in range(0, len(L) - 10, 10):
        sessions.append(L[s:s + 10])                                                         # 続いた 10 話 = 1 つの読み
random.Random(0).shuffle(sessions); sessions = sessions[:40]
CAPS = [20, 50, 200, None]; POL = ["新しさ", "回数", "光の強さ", "自覚 × 光の強さ", "2 段（短期 + 中期）"]


class Memory:
    """名札の記憶。policy に従って、上限 cap を超えたら点の低い名札を捨てる。"""
    def __init__(self, policy, cap):
        self.p, self.cap = policy, cap
        self.cnt = collections.Counter(); self.last = {}; self.str = collections.defaultdict(float); self.sur = {}
        self.mid = collections.defaultdict(float); self.t = 0

    def tick(self):                                                                          # 1 文進む: 光が薄まる
        self.t += 1
        for k in list(self.str):
            self.str[k] *= 0.97
        for k in list(self.mid):
            self.mid[k] *= 0.995

    def add(self, w, surprise):
        self.cnt[w] += 1; self.last[w] = self.t; self.str[w] += 1.0; self.sur.setdefault(w, surprise)
        if self.p == "2 段（短期 + 中期）" and w in self.mid:
            self.mid[w] += 1.0
        self.evict()

    def score(self, w):
        return {"新しさ": self.last[w], "回数": self.cnt[w], "光の強さ": self.str[w], "自覚 × 光の強さ": self.sur[w] * self.str[w],
                "2 段（短期 + 中期）": self.str[w]}[self.p]

    def evict(self):
        cap = None if self.cap is None else (self.cap // 4 if self.p == "2 段（短期 + 中期）" else self.cap)
        while cap is not None and len(self.cnt) > cap:
            w = min(self.cnt, key=self.score)
            if self.p == "2 段（短期 + 中期）" and self.cnt[w] >= 2:                         # 2 回以上出た名札は中期へ
                self.mid[w] += self.str[w]
                while len(self.mid) > 3 * self.cap // 4:
                    del self.mid[min(self.mid, key=self.mid.get)]
            for dct in (self.cnt, self.last, self.str):
                dct.pop(w, None)

    def dist(self):
        """名札の記憶の答え P(語)（回数や強さの割合）。"""
        p = torch.zeros(len(WORDS))
        for w in self.cnt:
            p[WI[w]] += self.cnt[w] if self.p in ("新しさ", "回数") else self.str[w]
        for w, v in self.mid.items():
            p[WI[w]] += v
        return p / p.sum() if p.sum() > 0 else p


st = collections.defaultdict(lambda: [0.0, 0]); mids = collections.defaultdict(list); n_q = 0
for sess in sessions:
    sents = [x for d in sess for t in d["text"].split("\n") for x in re.split(r"(?<=[。！？])", t) if len(x.strip()) > 1]
    NS = [nouns(s) for s in sents]
    mems = {(p, c): Memory(p, c) for p in POL for c in CAPS if not (c is None and p != "回数")}   # 上限なしは 1 つだけ（どの基準でも同じ）
    for k in range(len(NS)):
        if k >= 10 and NS[k] and NS[k - 1] and k % 3 == 0:                                   # 3 文に 1 回測る（時間を節約）
            base = LOGB[[WI[w] for w in NS[k - 1]]].sum(0)[LAB] + lpw; pl = torch.softmax(base, 0)
            seen = set(w for s in NS[:k] for w in s)
            for key, m in mems.items():
                s = torch.log(0.3 * pl + 0.7 * m.dist() + 1e-30)
                for w in set(NS[k]):
                    r = int((s > s[WI[w]]).sum())
                    for kind in ("全部", "くり返し" if w in seen else "初めて"):
                        st[(key, kind)][0] += 1 / (r + 1); st[(key, kind)][1] += 1
            n_q += len(set(NS[k]))
        if NS[k]:
            base_k = LOGB[[WI[w] for w in NS[k - 1]]].sum(0)[LAB] + lpw if k > 0 and NS[k - 1] else lpw
            lp = torch.log_softmax(base_k, 0)
            for key, m in mems.items():
                m.tick()
                for w in NS[k]:
                    m.add(w, float(-lp[WI[w]]))                                              # 意外さ = −log P(語 | 長期)
    for key, m in mems.items():
        mids[key].append(len(m.mid))
res = dict(exp="81-3", sessions=len(sessions), questions=n_q, sec=time.time() - t0,
           table={f"{p}|上限 {c}|{kind}": dict(MRR=round(v[0] / v[1], 4), n=v[1]) for ((p, c), kind), v in st.items()},
           mid_kept={f"{p}|上限 {c}": sum(v) / len(v) for (p, c), v in mids.items() if p == "2 段（短期 + 中期）"})
with open("results/exp81.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
for k in sorted(res["table"]):
    if "|全部" in k or "くり返し" in k:
        print(k, res["table"][k])
print("中期に残った名札（長期へ移す候補）", res["mid_kept"])
