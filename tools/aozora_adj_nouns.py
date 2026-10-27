"""青空文庫から「形容詞 + 名詞」の組を広く数える（大規模実験 B-1 の材料）

tools/aozora_pairs.py は物の名前を 12 個に決めて数えた。ここでは名詞を決めずに、形容詞のすぐ後の漢字 1〜3 文字の言葉を全部数え、
よく出る名詞を選ぶ（形態素解析は使わない。「長い言葉の一部」を避けるため、後ろに漢字・カタカナが続くものは数えない）。
観点 9 つ: 色・大きさ・温度・味・長さ・高さ・重さ・明るさ・新しさ。
材料: Project_Kotoyui が取得した青空文庫（著作権の切れた作品）。文章はリポジトリに置かず、数えた結果だけを results/aozora_adj_nouns.json に。
実行: python tools/aozora_adj_nouns.py
"""
import collections, glob, json, re, sys, zipfile

sys.stdout.reconfigure(encoding="utf-8")
VIEW = {"色": {"赤": ["赤い", "紅い", "真赤な", "真っ赤な"], "白": ["白い", "真白な", "真っ白な"], "黒": ["黒い", "真黒な", "真っ黒な"],
               "青": ["青い", "蒼い", "碧い"], "黄": ["黄色い", "黄いろい"]},
        "大きさ": {"大": ["大きな", "大きい"], "小": ["小さな", "小さい"]},
        "温度": {"冷たい": ["冷たい", "つめたい", "冷い"], "温かい": ["暖かい", "温かい", "あたたかい", "暖い"], "熱い": ["熱い"]},
        "味": {"甘い": ["甘い"], "酸っぱい": ["酸っぱい", "すっぱい"], "苦い": ["苦い"]},
        "長さ": {"長い": ["長い"], "短い": ["短い"]}, "高さ": {"高い": ["高い"], "低い": ["低い"]},
        "重さ": {"重い": ["重い"], "軽い": ["軽い"]}, "明るさ": {"明るい": ["明るい"], "暗い": ["暗い"]},
        "新しさ": {"新しい": ["新しい"], "古い": ["古い"]}}
forms = {f: (v, val) for v, vals in VIEW.items() for val, fs in vals.items() for f in fs}
pat = re.compile("(" + "|".join(sorted(map(re.escape, forms), key=len, reverse=True)) + r")([一-龥々]{1,3})(?![一-龥々ァ-ヶ])")
count = collections.defaultdict(lambda: collections.defaultdict(collections.Counter)); chars = 0
files = sorted(glob.glob("D:/AI_git/Project_Kotoyui/data/aozora/raw/*.zip"))
for i, f in enumerate(files):
    try:
        with zipfile.ZipFile(f) as z:
            for name in z.namelist():
                if name.endswith(".txt"):
                    t = re.sub(r"《[^》]*》|［＃[^］]*］|｜", "", z.read(name).decode("cp932", errors="ignore")); chars += len(t)
                    for m in pat.finditer(t):
                        v, val = forms[m.group(1)]; count[m.group(2)][v][val] += 1
    except zipfile.BadZipFile:
        pass
tot = {n: sum(sum(c.values()) for c in d.values()) for n, d in count.items()}
nviews = {n: sum(1 for c in d.values() if sum(c.values()) >= 3) for n, d in count.items()}
top = [n for n in sorted(tot, key=tot.get, reverse=True) if nviews[n] >= 3][:300]
json.dump(dict(source="青空文庫（Project_Kotoyui/data/aozora/raw、%d 作品、約 %d 文字）" % (len(files), chars), VIEW={v: list(x) for v, x in VIEW.items()},
               nouns=top, count={n: {v: dict(count[n][v]) for v in VIEW} for n in top}, total={n: tot[n] for n in top}),
          open("results/aozora_adj_nouns.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print("名詞（観点 3 つ以上で 3 回以上、多い順）:", len(top))
print(" ".join(f"{n}{tot[n]}" for n in top[:120]))
