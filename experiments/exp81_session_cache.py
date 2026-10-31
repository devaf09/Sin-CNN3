"""実験81-1: 文章の中のキャッシュ（持ち主の第 3 案 3-3）— その文章に出た語を増幅すると、後で出る語が当たるか

着想: 2026-10-31 持ち主「1 文章単位中に出てくる単語に、属性からの単語選択のバイアスか増幅を持たせる。
      『アップル』を強く学習すると基本の知識が偏るので、1 文章内では差分学習のような仕組みで、光で融合して結果を変える」。docs/grammar.md 6 節。
      言語モデルの研究で「キャッシュ」と呼ばれる考え（いま読んでいる文章に出た語は、また出やすい）と同じ。
問題: 持ち主が集めた Web 小説（実験だけ）の確かめ用 14 作品（学習に使っていない作品）の話。各話の 10 文目から先の文について、
      その文の名詞（語板 v3 の 3 万語）を、それより前の文だけから当てる（3 万語の中の順位）。
長期（語板）の答え: 隣の文の観点の語板（実験77、web2）で、すぐ前の文の名詞から P(仲間 | 語) を足したもの + log(語の多さ)。
キャッシュ: その話の中で、それまでに出た語の回数 n を、log(1 + n) × 強さ β で足す（光で言えば、文章の観点の明るさを融合で足す）。
      覚えておく長さ: 話の始めから全部 / 直前 50 文 / 直前 10 文。強さ β = 0・0.5・1・2・4。
測るもの: MRR（3 万語の中の順位の逆数の平均）と上位 10 に入る割合。当てたい語を「前に出た語（くり返し）」と「初めて出る語」に分けても測る
      （キャッシュはくり返しを助け、初めての語は邪魔するはず）。
測り方の限界: 当てているのは「その語が出るか」だけで、意味が合っているかは測らない。否定（「アップルではない」）は逆になるが区別しない。
結果: results/exp81.txt（数だけ）。実行: python experiments/exp81_session_cache.py
"""
import collections, json, math, os, random, re, sys, time, zlib

import torch

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t0 = time.time()
PRIV = "D:/AI_git/Sin-CNN3_private"
NX = torch.load(f"{PRIV}/models/goban_next_web2.pt", weights_only=False)                    # 実験77 の語板（語の並び = 語板 v3）
WORDS = NX["words"]; WI = {w: i for i, w in enumerate(WORDS)}; LAB = NX["cluster"]
LOGB = NX["board"].clamp(min=1e-9).log()                                                    # (語, 仲間 + その他) 語板の答えの log
import pickle
FREQ = pickle.load(open(f"{PRIV}/data/views/merged2.pkl", "rb"))["noun_total"]
lpw = torch.tensor([math.log(1 + FREQ.get(w, 0)) for w in WORDS])
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").tokenizer(); MODE = SplitMode.C
nouns = lambda s: [m.normalized_form() for m in tok.tokenize(s[:500], MODE) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("普通名詞", "固有名詞") and m.normalized_form() in WI]
docs = [d for d in map(json.loads, open(f"{PRIV}/data/text/webnovels2.jsonl", encoding="utf-8")) if zlib.crc32(d["work"].encode()) % 10 == 0]
random.Random(0).shuffle(docs); docs = docs[:300]                                             # 300 話
MODE2 = os.environ.get("MIX") == "1"                                                        # MIX=1: 足し算の強さを広げ、確率の混ぜ合わせ（キャッシュの言語モデルの形）も比べる
BETAS = [0, 4, 8, 16] if MODE2 else [0, 0.5, 1, 2, 4]
WINS = {"全部": 10 ** 9} if MODE2 else {"全部": 10 ** 9, "直前 50 文": 50, "直前 10 文": 10}
LAMS = [0.1, 0.3, 0.5, 0.7] if MODE2 else []
st = collections.defaultdict(lambda: [0.0, 0.0, 0])                                         # (設定, 区分) → [MRR の和, 上位 10 の数, 問題の数]
n_q = 0
for d in docs:
    sents = [x for t in d["text"].split("\n") for x in re.split(r"(?<=[。！？])", t) if len(x.strip()) > 1]
    NS = [nouns(s) for s in sents]
    for k in range(10, len(NS)):
        targets = set(NS[k])
        if not targets or not NS[k - 1]:
            continue
        prev = [WI[w] for w in NS[k - 1]]
        base = LOGB[prev].sum(0)[LAB] + lpw                                                  # 長期の答え（隣の文の観点 + 語の多さ）
        seen = {name: collections.Counter(w for s in NS[max(0, k - L):k] for w in s) for name, L in WINS.items()}
        for w in targets:
            j = WI[w]; kind = "くり返し" if seen["全部"][w] > 0 else "初めて"
            for name, cnt in seen.items():
                cache = torch.zeros(len(WORDS))
                for x, n in cnt.items():
                    cache[WI[x]] = math.log(1 + n)
                if LAMS:                                                                     # 混ぜ合わせ: P = (1 − λ) P(長期) + λ P(文章に出た語の回数の割合)
                    pl = torch.softmax(base, 0); pc = cache.exp() - 1; pc = pc / pc.sum().clamp(min=1e-9)
                    for lam in LAMS:
                        s = torch.log((1 - lam) * pl + lam * pc + 1e-30); r = int((s > s[j]).sum())
                        for key in ((name, f"混ぜ λ={lam}", kind), (name, f"混ぜ λ={lam}", "全部")):
                            st[key][0] += 1 / (r + 1); st[key][1] += r < 10; st[key][2] += 1
                for b in BETAS:
                    if b == 0 and name != "全部":
                        continue                                                             # β = 0 は覚えておく長さによらない
                    s = base + b * cache
                    r = int((s > s[j]).sum())
                    for key in ((name, b, kind), (name, b, "全部")):
                        st[key][0] += 1 / (r + 1); st[key][1] += r < 10; st[key][2] += 1
            n_q += 1
    if n_q > 6000:
        break
res = dict(exp="81-1b" if MODE2 else "81-1", questions=n_q, docs_used=len(docs), sec=time.time() - t0,
           table={f"{n}|β={b}|{k}": dict(MRR=round(v[0] / v[2], 4), top10=round(v[1] / v[2], 4), n=v[2]) for (n, b, k), v in st.items()})
with open("results/exp81.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
for key in sorted(res["table"]):
    print(key, res["table"][key])
