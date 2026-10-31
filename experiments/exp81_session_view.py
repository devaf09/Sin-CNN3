"""実験81-2: 文章の観点（持ち主の第 3 案 3-2）— 文章に出た語の「属性の光」を融合した観点と、語そのものの名札（キャッシュ）を比べる

着想: 持ち主「観点周波数を増やして、1 文章単位の間だけ書き換わる共通属性の観点を持つ。新たに出てきた語板には、その共通部分の属性と名札を紐づける」
      「敵 + 倒す + 呪文 で『ザキラ』が既に語板としてある場合が問題になるかも」。
比べるもの（実験81-1 と同じ問題: Web 小説の確かめ用 14 作品の 300 話、各文の名詞を前の文だけから当てる）:
  長期だけ / 名札（語そのもの）のキャッシュ（実験81-1 の混ぜ合わせ λ = 0.7）/
  属性の光の観点: それまでに出た語の語板 v3 の属性の明るさ（405）を融合（明るさの平均）し、候補の語板との重なり（cos）を足す（β = 1・2・4）/
  名札 + 属性の光。
  「似た語に引っ張られた」割合: 1 位が外れた問題のうち、1 位の語が文章に出ていない、かつ文章に出た語のどれかと属性がよく似ている（cos 0.9 以上）割合。
  （ザキラの形 = 長期の語板では同じ属性を持つ別の語）
測り方の限界: 実験81-1 と同じ（語が出るかだけ。意味・否定は見ない）。
結果: results/exp81.txt。実行: python experiments/exp81_session_view.py
"""
import collections, json, math, os, pickle, random, re, sys, time, zlib

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t0 = time.time()
PRIV = "D:/AI_git/Sin-CNN3_private"
NX = torch.load(f"{PRIV}/models/goban_next_web2.pt", weights_only=False)
WORDS = NX["words"]; WI = {w: i for i, w in enumerate(WORDS)}; LAB = NX["cluster"]; LOGB = NX["board"].clamp(min=1e-9).log()
Ps = [torch.load(f"{PRIV}/models/goban_v3_part{k}_scratch.pt", weights_only=False)["P"] for k in (0, 1)]
with torch.no_grad():
    P = torch.cat(Ps); B3 = torch.cat([run(P[i:i + 1500], 9, 54) for i in range(0, len(WORDS), 1500)])
FL = torch.nn.functional.normalize(((B3 / B3.sum(-1, keepdim=True))[:, :, :45]).reshape(len(WORDS), -1), dim=1)   # 語ごとの属性の明るさ（長さ 1 にそろえる）
FREQ = pickle.load(open(f"{PRIV}/data/views/merged2.pkl", "rb"))["noun_total"]
lpw = torch.tensor([math.log(1 + FREQ.get(w, 0)) for w in WORDS])
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").tokenizer(); MODE = SplitMode.C
nouns = lambda s: [m.normalized_form() for m in tok.tokenize(s[:500], MODE) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("普通名詞", "固有名詞") and m.normalized_form() in WI]
docs = [d for d in map(json.loads, open(f"{PRIV}/data/text/webnovels2.jsonl", encoding="utf-8")) if zlib.crc32(d["work"].encode()) % 10 == 0]
random.Random(0).shuffle(docs); docs = docs[:300]
z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
st = collections.defaultdict(lambda: [0.0, 0.0, 0]); pulled = collections.Counter(); wrong = collections.Counter(); n_q = 0
for d in docs:
    sents = [x for t in d["text"].split("\n") for x in re.split(r"(?<=[。！？])", t) if len(x.strip()) > 1]
    NS = [nouns(s) for s in sents]
    for k in range(10, len(NS)):
        targets = set(NS[k])
        if not targets or not NS[k - 1]:
            continue
        base = LOGB[[WI[w] for w in NS[k - 1]]].sum(0)[LAB] + lpw
        cnt = collections.Counter(w for s in NS[:k] for w in s); ids = [WI[w] for w in cnt]
        pc = torch.zeros(len(WORDS)); pc[ids] = torch.tensor([float(cnt[w]) for w in cnt]); pc /= pc.sum()
        sess = (FL[ids] * torch.tensor([float(cnt[w]) for w in cnt])[:, None]).sum(0)          # 文章の観点 = 出た語の属性の光の融合
        s_attr = z(FL @ torch.nn.functional.normalize(sess, dim=0))
        pl = torch.softmax(base, 0)
        S = {"長期だけ": base, "名札のキャッシュ（λ=0.7）": torch.log(0.3 * pl + 0.7 * pc + 1e-30)}
        for b in (1, 2, 4):
            S[f"属性の光の観点（β={b}）"] = base + b * s_attr
        S["名札 + 属性の光（β=2）"] = torch.log(0.3 * torch.softmax(base + 2 * s_attr, 0) + 0.7 * pc + 1e-30)
        simmax = (FL @ FL[ids].T).max(1).values                                              # 文章に出た語との似かたのいちばん大きい値
        for w in targets:
            j = WI[w]; kind = "くり返し" if cnt[w] > 0 else "初めて"
            for name, s in S.items():
                r = int((s > s[j]).sum())
                for key in ((name, kind), (name, "全部")):
                    st[key][0] += 1 / (r + 1); st[key][1] += r < 10; st[key][2] += 1
                if r > 0:
                    t1 = int(s.argmax()); wrong[name] += 1
                    pulled[name] += (cnt[WORDS[t1]] == 0) and float(simmax[t1]) >= 0.9         # 文章に出ていない、似た語に引っ張られた
            n_q += 1
    if n_q > 6000:
        break
res = dict(exp="81-2", questions=n_q, sec=time.time() - t0,
           table={f"{n}|{k}": dict(MRR=round(v[0] / v[2], 4), top10=round(v[1] / v[2], 4), n=v[2]) for (n, k), v in st.items()},
           pulled={n: round(pulled[n] / max(1, wrong[n]), 3) for n in wrong})
with open("results/exp81.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
for k in sorted(res["table"]):
    print(k, res["table"][k])
print("似た語に引っ張られた割合（1 位が外れた問題のうち）", res["pulled"])
