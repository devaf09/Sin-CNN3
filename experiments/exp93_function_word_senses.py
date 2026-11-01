"""実験93: 同じ文字の機能語を、働きごとに別の板に分ける（持ち主の案 2026-11-02）

着想: 持ち主「文字列が同じだからその役割も同じにはならない（りんごの例みたいな）。品詞であっても、属する語群に重みを持たせた、複数の役割の同名の品詞も存在する。
      その場合は語板のように分けて作る必要がある」。
方法（ラベルを使わない）: 同じ文字の語（で・が・と・に・の・から・ない・て・も・な）の出てきた場面ごとに、前後 2 つずつの品詞（大分類 + 句読点 + 文の始め・終わり）を
      並べた「場面の光」を作り、k 平均法で 2〜4 の仲間に分ける（点 = 仲間の中の近さ − 仲間どうしの近さ、がいちばん良い k を選ぶ）。仲間 1 つ = 1 枚の板。
      分けたあとで、辞書（Sudachi）の品詞（例: で = 格助詞 / 接続助詞 / 助動詞「だ」の連用形、ない = 助動詞 / 形容詞「無い」、が = 格助詞 / 接続助詞）と比べる（純度）。
      板ごとの「求めるもの」（実験91 と同じ差分）も出す。
材料: 実験84 と同じ（会話・青空文庫 300 作品・Web 小説 3000 話）から、語ごとに 2 万の場面。
結果: results/exp93.txt、docs/function_word_senses_93.md。実行: python experiments/exp93_function_word_senses.py
"""
import collections, json, os, pickle, random, sys

import torch

sys.stdout.reconfigure(encoding="utf-8"); torch.manual_seed(0); PRIV = "D:/AI_git/Sin-CNN3_private"
DOCS = pickle.load(open(f"{PRIV}/data/views/synth_units.pkl", "rb"))
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").tokenizer(); MODE = SplitMode.C
WORDS = ["で", "が", "と", "に", "の", "から", "ない", "て", "も", "な"]


def coarse(p):
    if p[0] == "補助記号":
        return "句点" if p[1] == "句点" else "読点" if p[1] == "読点" else "記号"
    return p[0]


# 場面: 元の文を Sudachi で語に分け直し、表層が対象の語の所を集める（品詞は比べるためだけに残す）
lines = []
for s, rows in DOCS:
    for r in rows:
        lines.append("".join(w for w, _ in r))
random.Random(0).shuffle(lines)
occ = collections.defaultdict(list)
for line in lines[:400000]:
    ms = tok.tokenize(line[:200], MODE); ps = [coarse(m.part_of_speech()) for m in ms]
    for i, m in enumerate(ms):
        w = m.surface()
        if w in WORDS and len(occ[w]) < 20000:
            ctx = [("始め" if i + d < 0 else "終わり" if i + d >= len(ms) else ps[i + d]) for d in (-2, -1, 1, 2)]
            p = m.part_of_speech(); gold = p[0] + ("-" + p[1] if p[1] != "*" else "") + ("（" + m.dictionary_form() + "）" if p[0] in ("助動詞", "形容詞") else "")
            nxt = ms[i + 1].surface() if i + 1 < len(ms) else "終わり"
            occ[w].append((ctx, gold, nxt))
    if all(len(occ[w]) >= 20000 for w in WORDS):
        break
CLS = sorted({c for w in WORDS for ctx, _, _ in occ[w] for c in ctx}); CI = {c: i for i, c in enumerate(CLS)}; NC = len(CLS)


def vec(ctx):
    v = torch.zeros(4 * NC)
    for k, c in enumerate(ctx):
        v[k * NC + CI[c]] = 1.0 if k in (1, 2) else 0.5                                      # すぐ前・すぐ後を重く
    return v


def kmeans(X, k, it=40):
    Xn = torch.nn.functional.normalize(X, dim=1); g = torch.Generator().manual_seed(0); C = Xn[torch.randperm(len(Xn), generator=g)[:k]].clone()
    for _ in range(it):
        lab = (Xn @ C.T).argmax(1)
        for c in range(k):
            if (lab == c).any():
                C[c] = torch.nn.functional.normalize(Xn[lab == c].mean(0), dim=0)
    within = float((Xn * C[lab]).sum(1).mean()); between = float((C @ C.T)[~torch.eye(k, dtype=torch.bool)].mean()) if k > 1 else 0
    return lab, within - between, C


res = dict(exp="93", words={}); L = ["# 同じ文字の機能語を働きごとに分ける（実験93）", "",
                                     "持ち主の案（2026-11-02）。作り方: experiments/exp93_function_word_senses.py。前後 2 つずつの品詞（場面の光）だけで、ラベルを使わずに 2〜4 の板に分け、あとで辞書の品詞と比べた。", ""]
for w in WORDS:
    X = torch.stack([vec(c) for c, _, _ in occ[w]]); gold = [g for _, g, _ in occ[w]]
    best = max(((kmeans(X, k), k) for k in (2, 3, 4)), key=lambda t: t[0][1])
    (lab, sc, C), k = best
    pur = sum(collections.Counter(gold[i] for i in range(len(gold)) if int(lab[i]) == c).most_common(1)[0][1] for c in range(k) if (lab == c).any()) / len(gold)
    base = collections.Counter(gold).most_common(1)[0][1] / len(gold)                         # 分けずに 1 枚のとき（いちばん多い品詞の割合）
    res["words"][w] = dict(k=k, purity=pur, one_board=base, gold=dict(collections.Counter(gold).most_common(5)))
    L += [f"## 「{w}」: {k} 枚に分かれた（辞書の品詞との一致 {pur:.2f}、分けない 1 枚なら {base:.2f}）", "", "| 板 | 場面の数 | 辞書の品詞（多い順） | 求めるもの（前・後で多い品詞） | すぐ後の語の例 |", "|---|---|---|---|---|"]
    for c in range(k):
        ids = [i for i in range(len(gold)) if int(lab[i]) == c]
        if not ids:
            continue
        gc = collections.Counter(gold[i] for i in ids).most_common(3)
        prev = collections.Counter(occ[w][i][0][1] for i in ids).most_common(2); nxt = collections.Counter(occ[w][i][0][2] for i in ids).most_common(2)
        nw = collections.Counter(occ[w][i][2] for i in ids).most_common(5)
        L.append(f"| {w}#{c + 1} | {len(ids)} | {'・'.join(f'{g} {m / len(ids):.0%}' for g, m in gc)} | 前 {'・'.join(p for p, _ in prev)} ／ 後 {'・'.join(p for p, _ in nxt)} | {'・'.join(x for x, _ in nw)} |")
    L.append("")
with open("results/exp93.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
open("docs/function_word_senses_93.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
