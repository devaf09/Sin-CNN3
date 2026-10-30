"""実験76 の「上限」用の表: 光を使わず、ふつうの数え方で「隣の文に一緒に出た名詞」を数える（持ち主 2026-10-30「実験以外に確かめようはないか」への案）。
B と D から C を当てる問題で、数え方の表がどこまで当てられるか（上限の目安）を、実験76 と同じ測り方で比べるため。
表: N[x][y] = 名詞 x を含む文の、すぐ前かすぐ後の文に、名詞 y が出た回数（語板 v3 の 3 万語だけ）。
材料: SOURCE=aozora → 青空文庫のうち実験76 の問題に使わない作品から 600 作品、SOURCE=conv → 会話（RealPersonaChat・JMultiWOZ）の学習用の側。
出力: 非公開フォルダの data/views/adjacent_<SOURCE>.pkl。実行: SOURCE=aozora python experiments/exp76_ceiling_table.py
"""
import collections, json, os, pickle, random, re, sys, time, zlib

import torch
from sudachipy import Dictionary, SplitMode

sys.stdout.reconfigure(encoding="utf-8"); t0 = time.time()
PRIV = "D:/AI_git/Sin-CNN3_private"; SOURCE = os.environ.get("SOURCE", "aozora")
WI = set(w for k in (0, 1) for w in torch.load(f"{PRIV}/models/goban_v3_part{k}_scratch.pt", weights_only=False)["words"])
tok = Dictionary(dict="core").tokenizer(); MODE = SplitMode.C
nouns = lambda s: {m.normalized_form() for m in tok.tokenize(s[:500], MODE) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "普通名詞" and m.normalized_form() in WI}
if SOURCE == "conv":
    docs = [d for d in map(json.loads, open(f"{PRIV}/data/text/conv_all.jsonl", encoding="utf-8"))
            if not d["id"].startswith("aozora") and zlib.crc32(f"conv:{d['id']}".encode()) % 10 != 0]       # 確かめ用（% 10 == 0）は使わない
    split = lambda d: [x.strip() for x in d["text"].split("\n") if x.strip()]
else:
    docs = [json.loads(l) for l in open(f"{PRIV}/data/text/aozora_full.jsonl", encoding="utf-8")]
    rnd = random.Random(0); test = [d for d in docs if 30000 < len(d["text"]) < 300000]; rnd.shuffle(test)
    used = set(d["id"] for d in test[:60])                                                  # 実験76 の問題に使った 60 作品は除く
    rest = [d for d in docs if d["id"] not in used and len(d["text"]) > 5000]; random.Random(1).shuffle(rest); docs = rest[:int(os.environ.get("NDOC", "600"))]                # NDOC: 使う作品の数（実験77 は全部）
    split = lambda d: [s for s in re.split(r"(?<=。)", d["text"].replace("\r", "").replace("\n", "").replace("　", "")) if len(s) > 8]
N = collections.defaultdict(collections.Counter); TOT = collections.Counter()
for i, d in enumerate(docs):
    NS = [nouns(s) for s in split(d)]
    for k in range(len(NS)):
        for j in (k - 1, k + 1):                                                            # すぐ前・すぐ後の文
            if 0 <= j < len(NS):
                for x in NS[k]:
                    for y in NS[j]:
                        if y != x:
                            N[x][y] += 1
        TOT.update(NS[k])
    if i % 100 == 0:
        print(i, len(docs), f"{time.time() - t0:.0f} 秒", flush=True)
pickle.dump(dict(N={x: dict(c) for x, c in N.items()}, TOT=dict(TOT), source=SOURCE, docs=len(docs)), open(f"{PRIV}/data/views/adjacent_{SOURCE}.pkl", "wb"))
print("おわり", len(N), f"{time.time() - t0:.0f} 秒")
