"""実験81-1c: 非線形のキャッシュ（持ち主 2026-10-31「3-3 のキャッシュに非線形の特性を持たせることは出来る？」）

実験81-1 と同じ問題（Web 小説の確かめ用 14 作品の 300 話、各文の名詞を前の文だけから 3 万語の中で当てる、6,282 問）。
比べるもの（いずれも P = (1 − λ) P(長期) + λ P(名札)、P(名札) ∝ f(回数)）:
  線形（実験81-1 の λ = 0.7、f = 回数）
  しきい値: f = 回数 × シグモイド(回数 − θ)（θ = 1.5・2.5。1 回だけ出た語は弱く、θ を超えると急に強く）
  べき: f = 回数^p（p = 0.5・2）
  長期の自信で切り替え: λ = λ_max × （長期の答えの平たさ = エントロピー ÷ 最大）^q（自信がないほどキャッシュを強く。λ_max = 0.9、q = 1・3）
測り方の限界: 語が出るかだけ。意味・否定は見ない。結果: results/exp81.txt。実行: python experiments/exp81_cache_nonlinear.py
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
docs = [d for d in map(json.loads, open(f"{PRIV}/data/text/webnovels2.jsonl", encoding="utf-8")) if zlib.crc32(d["work"].encode()) % 10 == 0]
random.Random(0).shuffle(docs); docs = docs[:300]
HMAX = math.log(len(WORDS))
st = collections.defaultdict(lambda: [0.0, 0.0, 0]); n_q = 0
for d in docs:
    sents = [x for t in d["text"].split("\n") for x in re.split(r"(?<=[。！？])", t) if len(x.strip()) > 1]
    NS = [nouns(s) for s in sents]
    for k in range(10, len(NS)):
        targets = set(NS[k])
        if not targets or not NS[k - 1]:
            continue
        base = LOGB[[WI[w] for w in NS[k - 1]]].sum(0)[LAB] + lpw; pl = torch.softmax(base, 0)
        flat = float(-(pl * torch.log(pl + 1e-30)).sum()) / HMAX                              # 長期の答えの平たさ（0 = はっきり、1 = まったく平ら）
        cnt = collections.Counter(w for s in NS[:k] for w in s)
        if k == 10 or "FIRST" not in dir():
            FIRST = torch.full((len(WORDS),), -10 ** 6, dtype=torch.float); SUR = torch.zeros(len(WORDS)); DONE = set()
        for kk in range(0 if k == 10 else k - 1, k):                                          # 初めて出た文の番号と、そのときの長期にとっての意外さ
            if kk > 0 and NS[kk - 1]:
                lpk = torch.log_softmax(LOGB[[WI[w] for w in NS[kk - 1]]].sum(0)[LAB] + lpw, 0)
            else:
                lpk = torch.log_softmax(lpw, 0)
            for w in NS[kk]:
                if w not in DONE:
                    DONE.add(w); FIRST[WI[w]] = kk; SUR[WI[w]] = float(-lpk[WI[w]])
        n = torch.zeros(len(WORDS)); n[[WI[w] for w in cnt]] = torch.tensor([float(c) for c in cnt.values()])

        def mix(f, lam):
            pc = f / f.sum() if f.sum() > 0 else f
            return torch.log((1 - lam) * pl + lam * pc + 1e-30)
        S = {"線形（λ=0.7）": mix(n, 0.7)}
        for th in (1.5, 2.5):
            S[f"しきい値 θ={th}"] = mix(n * torch.sigmoid(4 * (n - th)), 0.7)
        for p in (0.5, 2):
            S[f"べき p={p}"] = mix(n ** p, 0.7)
        for q in (1, 3):
            S[f"自信で切り替え q={q}"] = mix(n, 0.9 * flat ** q)
        # 逆特性（持ち主 2026-10-31「キャッシュ中に初めて出た単語ほど出やすくなる」）
        seen_ids = n > 0
        S["逆: 出た語はみな同じ（p=0）"] = mix(seen_ids.float(), 0.7)
        S["逆: 回数が少ないほど強く（p=−0.5）"] = mix(torch.where(seen_ids, n.clamp(min=1) ** -0.5, torch.zeros(1)), 0.7)
        S["逆: 長期にとって意外なほど強く"] = mix(torch.where(seen_ids, SUR, torch.zeros(1)), 0.7)
        S["逆: 新しく加わった語ほど強く"] = mix(torch.where(seen_ids, 0.97 ** (k - FIRST), torch.zeros(1)), 0.7)
        for w in targets:
            j = WI[w]; kind = "くり返し" if cnt[w] > 0 else "初めて"
            for name, s in S.items():
                r = int((s > s[j]).sum())
                for key in ((name, kind), (name, "全部")):
                    st[key][0] += 1 / (r + 1); st[key][1] += r < 10; st[key][2] += 1
            n_q += 1
    if n_q > 6000:
        break
res = dict(exp="81-1c", questions=n_q, sec=time.time() - t0,
           table={f"{n}|{k}": dict(MRR=round(v[0] / v[2], 4), top10=round(v[1] / v[2], 4), n=v[2]) for (n, k), v in st.items()})
with open("results/exp81.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
for k in sorted(res["table"]):
    print(k, res["table"][k])
