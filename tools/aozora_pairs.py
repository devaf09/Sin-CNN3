"""青空文庫の文から「形容詞 + 物の名前」の組を数える（実験25 の材料）

持ち主の案（2026-10-27）「光点は周囲の文脈や映像などからそれを学習する。例: 色の周波数選択: 文章から色は赤 80%、緑 20% みたいな」。
文の中で、物の名前（猫・雪など）のすぐ前に来る形容詞（赤い・白い・大きな・冷たい・甘い など）を数え、観点ごとの割合にする。
材料: Project_Kotoyui が取得した青空文庫（著作権の切れた作品、自由に複製・再配布できる。D:/AI_git/Project_Kotoyui/data/aozora/raw/*.zip）。
      文章そのものはこのリポジトリに置かない（数えた結果だけを results/aozora_pairs.json に置く）。
実行: python tools/aozora_pairs.py
"""
import glob, json, re, sys, zipfile

sys.stdout.reconfigure(encoding="utf-8")
OBJ = ["猫", "犬", "船", "花", "雪", "月", "林檎", "蜜柑", "空", "海", "血", "火"]
VIEW = {"色": {"赤": ["赤い", "紅い", "真赤な", "真っ赤な", "赤き"], "白": ["白い", "真白な", "真っ白な", "白き"], "黒": ["黒い", "真黒な", "真っ黒な", "黒き"],
               "青": ["青い", "蒼い", "碧い", "青き"], "黄": ["黄色い", "黄いろい", "黄色な"]},
        "大きさ": {"大": ["大きな", "大きい"], "小": ["小さな", "小さい"]},
        "温度": {"冷たい": ["冷たい", "つめたい", "冷い"], "温かい": ["暖かい", "温かい", "あたたかい", "暖い"], "熱い": ["熱い", "あつい"]},
        "味": {"甘い": ["甘い"], "酸っぱい": ["酸っぱい", "酸い", "すっぱい"], "苦い": ["苦い"]}}
count = {o: {v: {val: 0 for val in vals} for v, vals in VIEW.items()} for o in OBJ}
# 物の名前のすぐ後に漢字・カタカナが続くもの（「空気」「火鉢」のような長い言葉）は数えない
pats = [(o, v, val, re.compile(re.escape(form) + re.escape(o) + r"(?![一-龥々ァ-ヶ])")) for o in OBJ for v, vals in VIEW.items() for val, forms in vals.items() for form in forms]
files = sorted(glob.glob("D:/AI_git/Project_Kotoyui/data/aozora/raw/*.zip")); chars = 0
for i, f in enumerate(files):
    try:
        with zipfile.ZipFile(f) as z:
            for name in z.namelist():
                if name.endswith(".txt"):
                    t = z.read(name).decode("cp932", errors="ignore")
                    t = re.sub(r"《[^》]*》|［＃[^］]*］|｜", "", t)                     # ふりがな・注記を除く
                    chars += len(t)
                    for o, v, val, p in pats:
                        count[o][v][val] += len(p.findall(t))
    except zipfile.BadZipFile:
        pass
    if i % 500 == 0:
        print(f"{i}/{len(files)}", flush=True)
json.dump(dict(source="青空文庫（Project_Kotoyui/data/aozora/raw、作品数 %d、約 %d 文字）" % (len(files), chars), VIEW={v: list(x) for v, x in VIEW.items()},
               forms=VIEW, count=count), open("results/aozora_pairs.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for o in OBJ:
    print(o, {v: {k: n for k, n in x.items() if n} for v, x in count[o].items()})
