"""実験117: 文の結びを、話者関係（丁寧・くだけた）に合わせて付け替える（テストボット02 の次、言結いの最初の目標）

考え（中身 + 尾、持ち主との話 2026-11-02）: 文の最後の文節を「中身（食べる）」と「尾（ました・た・ません・ない…）」に分け、尾だけを替える。
  1. 尾の意味の印: 過去（た）・否定（ない・ぬ・ん）・問い（か・の？）・誘い / 意志（う・よう・ましょう）・望み（たい）。丁寧さ（です・ます）は印に入れない。
  2. 教えの文の最後の文節から、（丁寧さ, 印, 中身の品詞）ごとに「よく出る尾」を数える。同じ中身の語で出た尾があればそれを優先。
  3. 中身の形（食べ・行っ・行き）は、（基本形, 尾の最初の語）でいちばん多かった形（実験105 と同じ）。
  付け替え = 印を保ったまま、相手の丁寧さの尾に替える。
物差し（人の採点なし）:
  往復: 丁寧 → くだけた → 丁寧 と付け替えて、元の文節に戻るか（尾を正しく分けて正しく選べていれば戻る）。
  印が保たれるか: 付け替えた文節を同じ方法で解析し直し、印（過去・否定・問い…）と丁寧さが狙いどおりか。
材料: Sin-CNN3_private/data/views/bunsetsu_roles.json の文の最後の文節（教え 19.8 万文節の中の ROOT、試しは別の文）。実験だけに使う材料を含むので、例は自分で書いた文だけを載せる。
結果: results/exp117.txt、docs/testbot02.md に追記。実行: python experiments/exp117_ending_swap.py
"""
import json, sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8"); PRIV = "D:/AI_git/Sin-CNN3_private"
FUNC = ("助詞", "助動詞", "補助記号")
NEG = {"ない", "ぬ", "ず", "ん", "無い"}


def split(words):
    """[(文字, 基本形, 品詞)] → 中身と尾（実験105 と同じ分け方）。"""
    i = 0
    while i < len(words) and not words[i][2].startswith(FUNC) and ("非自立" not in words[i][2] or i == 0):
        i += 1
    i = max(i, 1)
    return words[:i], words[i:]


def peel(tail):
    """尾の終わりの 終助詞（よ・ね・な…、か は問いの印なので残す）と記号を外す。外した部分は元のまま付け直す（話し方のくせ）。"""
    k = len(tail)
    while k > 0 and ((tail[k - 1][2].startswith("助詞-終助詞") and tail[k - 1][0] != "か") or (tail[k - 1][2].startswith("補助記号") and tail[k - 1][0] not in ("？", "?"))):   # ？ は問いの印なので残す
        k -= 1
    return tail[:k], tail[k:]


def marks(tail):
    """尾の意味の印（丁寧さは別）。"""
    L = [w[1] for w in tail]; O = [w[0] for w in tail]
    m = []
    if "た" in L:
        m.append("過去")
    if NEG & set(L):
        m.append("否定")
    if O and (O[-1] in ("か", "？", "?") or (len(O) >= 2 and O[-2] in ("か", "の") and O[-1] in ("？", "?"))):
        m.append("問い")
    if any(w[1] in ("う", "よう") and w[2].startswith("助動詞") for w in tail) or "ましょう" in "".join(O):
        m.append("誘い")                                                                      # 意志・誘いの う・よう（様子の「ようだ」は除く）
    if tail and tail[-1][2].startswith("助詞-接続助詞"):
        m.append("つなぎ")                                                                    # 〜て・〜けど・〜から で終わる（言いさし）
    if "たい" in L:
        m.append("望み")
    return tuple(m)


polite = lambda tail: any(w[1] in ("です", "ます") for w in tail)
rows = json.load(open(f"{PRIV}/data/views/bunsetsu_roles.json", encoding="utf-8"))
ends = {"train": [], "test": []}
for r in rows:
    if r["is_root"]:
        w = [(x[0], x[3], x[2]) for x in r["words"]]
        core, tail = split(w)
        tail, _ = peel(tail)
        if not tail or not core[-1][2].startswith(("動詞", "形容詞", "名詞", "形状詞")):
            continue
        ends[r["split"]].append((core, tail))
print("文の最後の文節 教え", len(ends["train"]), "試し", len(ends["test"]), flush=True)
TAB = defaultdict(Counter); TAB_L = defaultdict(Counter); FORM = defaultdict(Counter)
for core, tail in ends["train"]:
    lem = "".join(w[1] for w in core); pos = core[-1][2].split("-")[0]; t = tuple((w[0], w[1], w[2]) for w in tail)
    TAB[(polite(tail), marks(tail), pos)][t] += 1; TAB_L[(polite(tail), marks(tail), lem)][t] += 1
    FORM[(lem, tail[0][0])]["".join(w[0] for w in core)] += 1


def swap(core, tail, to_polite):
    """印を保って、丁寧さだけ替えた文節（文字）と、その尾。"""
    lem = "".join(w[1] for w in core); pos = core[-1][2].split("-")[0]; m = marks(tail)
    c = TAB_L.get((to_polite, m, lem)) or TAB.get((to_polite, m, pos))
    if not c:
        return None, None
    t = c.most_common(1)[0][0]
    f = FORM.get((lem, t[0][0]))
    stem = f.most_common(1)[0][0] if f else "".join(w[0] for w in core)
    return stem + "".join(w[0] for w in t), t


res = dict(exp="117")
ok_rt = ok_m = n = 0; by = Counter(); byn = Counter()
for core, tail in ends["test"][:20000]:
    p0 = polite(tail); orig = "".join(w[0] for w in core + tail)
    s1, t1 = swap(core, tail, not p0)
    if s1 is None:
        continue
    n += 1
    ok_m += marks(t1) == marks(tail) and polite(t1) == (not p0)                                  # 印と丁寧さが狙いどおり
    lem_core = [(w[0], w[1], w[2]) for w in core]
    s2, t2 = swap(lem_core, list(t1), p0)                                                       # 戻す
    good = s2 == orig; ok_rt += good; key = ("丁寧 → くだけ → 丁寧" if p0 else "くだけ → 丁寧 → くだけ") + " " + ("・".join(marks(tail)) or "印なし")
    by[key] += good; byn[key] += 1
res.update(n=n, roundtrip=ok_rt / n, marks_kept=ok_m / n, by={k: (round(by[k] / byn[k], 3), byn[k]) for k, _ in byn.most_common(12)})
print("付け替えた", n, "印と丁寧さが狙いどおり", round(ok_m / n, 3), "往復で元に戻る", round(ok_rt / n, 3), flush=True)
for k, v in res["by"].items():
    print(" ", k, v, flush=True)
# 例（自分で書いた文の最後の文節）: Sudachi で切って付け替える
from sudachipy import Dictionary
tok = Dictionary(dict="core").tokenizer()
ex = []
for s in ["昨日は駅前のラーメン屋に行きました", "明日は雨が降らないと思います", "一緒に映画を見ましょう", "もう宿題は終わったの？", "北海道に行きたい", "犬の散歩は大変だった", "まだ食べてない", "これは君の本ですか"]:
    ms = [(m.surface(), m.normalized_form(), "-".join(x for x in m.part_of_speech() if x != "*")) for m in tok.tokenize(s)]
    # 最後の自立語から文末までを最後の文節とみなす
    k = max(i for i, w in enumerate(ms) if not w[2].startswith(FUNC) and ("非自立可能" in w[2] or "非自立" not in w[2]) and not (i > 0 and ms[i - 1][2].startswith("助詞-接続助詞")))
    head, last = ms[:k], ms[k:]
    core, tail = split(last); tail, fin = peel(tail)
    out, t = swap(core, tail, not polite(tail))
    ex.append(dict(元=s, 付け替え=("".join(w[0] for w in head) + out + "".join(w[0] for w in fin)) if out else "（替えられない）", 向き="くだけた → 丁寧" if not polite(tail) else "丁寧 → くだけた"))
    print(ex[-1], flush=True)
res["examples"] = ex
with open("results/exp117.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(res, ensure_ascii=False) + "\n")
