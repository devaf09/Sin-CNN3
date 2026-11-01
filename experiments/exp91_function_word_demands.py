"""実験91: 品詞が「何を求められているか」を差分から取り出し、自立的に分ける（持ち主の案 2026-11-02）

着想: 持ち主「動詞も含めて、個数に上限のある品詞は、この特性の取得で小さな文法解析器ができあがるといい。わかる範囲で関連する品詞との差分から、
      その品詞が何を求められているかを特定して、ある程度自立的な学習を試してくれる？疑問 → 役割 → 否定はいい関連する流れ。これをすべての品詞で」。
方法（ラベルを使わない）:
  1. 語ごとの「前後に来る品詞」の割合（前 2・前 1・後 1・後 2 の位置、品詞の大分類 + 句読点 + 文の始め・終わり）を数える（実験82 の前後の品詞の語板と同じもの）。
  2. 差分 = その語の割合 − 同じ品詞の大分類（助詞・助動詞・動詞…）の平均の割合。差分が大きく正の所 = その語が特に「求める」もの、負の所 = 避けるもの。
  3. 差分だけで k 平均法で仲間分けし、辞書（Sudachi）の細かい分類（格助詞・係助詞・副助詞・接続助詞・終助詞・準体助詞・助動詞 / 動詞の自動・他動の目安）と比べる（純度）。
  語: 助詞・助動詞（よく出る 60 語、品詞つき）と、動詞（よく出る 200 語）。材料は実験84 と同じ（会話・青空文庫 300 作品・Web 小説 3000 話）。
  動詞の目安: 直前の助詞が「を」の割合が高いもの = 目的語をとる（他動）、が・に が高いもの = 自動、と仮に見て、仲間と比べる。
結果: results/exp91.txt、docs/function_word_demands_91.md。実行: python experiments/exp91_function_word_demands.py
"""
import collections, json, math, os, pickle, sys

import torch

sys.stdout.reconfigure(encoding="utf-8"); torch.manual_seed(0); PRIV = "D:/AI_git/Sin-CNN3_private"
DOCS = pickle.load(open(f"{PRIV}/data/views/synth_units.pkl", "rb"))                         # 行ごとの (語, 品詞の値)。品詞の値 = 助詞・助動詞は「助詞:が」、ほかは「大分類-中分類」
from sudachipy import Dictionary
tok = Dictionary(dict="core").tokenizer()


def coarse(l):
    """前後の位置に入れる品詞の粗い分類: 助詞・助動詞は語そのもの（よく出るものは区別したい）は使わず、大分類にする。句読点・括弧は別。"""
    if l.startswith("助詞:"):
        return "助詞"
    if l.startswith("助動詞:"):
        return "助動詞"
    if l.startswith("補助記号-句点"):
        return "句点"
    if l.startswith("補助記号-読点"):
        return "読点"
    if l.startswith("補助記号"):
        return "記号"
    return l.split("-")[0]


rows = [r for _, rs in DOCS for r in rs]
CLS = sorted({coarse(l) for r in rows[:200000] for _, l in r} | {"始め", "終わり"}); CI = {c: i for i, c in enumerate(CLS)}; NC = len(CLS)
POS = [-2, -1, 1, 2]
cnt = collections.defaultdict(lambda: torch.zeros(len(POS), NC)); freq = collections.Counter(); big = {}
for r in rows:
    labs = [coarse(l) for _, l in r]
    for t, (w, l) in enumerate(r):
        key = l if l.startswith(("助詞:", "助動詞:")) else (f"動詞:{w}" if l.startswith("動詞-一般") else None)
        if key is None:
            continue
        freq[key] += 1; big[key] = key.split(":")[0]
        for k, d in enumerate(POS):
            j = t + d
            c = "始め" if j < 0 else "終わり" if j >= len(r) else labs[j]
            cnt[key][k, CI[c]] += 1
fw = [k for k, _ in freq.most_common() if big[k] in ("助詞", "助動詞")][:60]
vb = [k for k, _ in freq.most_common() if big[k] == "動詞"][:200]
prof = {k: cnt[k] / cnt[k].sum(1, keepdim=True).clamp(min=1) for k in fw + vb}
mean = {g: torch.stack([prof[k] for k in ks]).mean(0) for g, ks in (("助詞", [k for k in fw if big[k] == "助詞"]), ("助動詞", [k for k in fw if big[k] == "助動詞"]), ("動詞", vb))}
diff = {k: prof[k] - mean[big[k]] for k in fw + vb}                                          # 差分 = 求めるもの（正）・避けるもの（負）


def demands(k, n=3):
    d = diff[k]; out = []
    for idx in torch.argsort(d.flatten(), descending=True)[:n]:
        p, c = divmod(int(idx), NC); out.append(f"{['前2', '前', '後', '後2'][p]}={CLS[c]} +{float(d[p, c]):.2f}")
    for idx in torch.argsort(d.flatten())[:1]:
        p, c = divmod(int(idx), NC); out.append(f"{['前2', '前', '後', '後2'][p]}={CLS[c]} {float(d[p, c]):.2f}")
    return "、".join(out)


def kmeans(X, k, it=50):
    X = torch.nn.functional.normalize(X, dim=1); g = torch.Generator().manual_seed(0); C = X[torch.randperm(len(X), generator=g)[:k]].clone()
    for _ in range(it):
        lab = (X @ C.T).argmax(1)
        for c in range(k):
            if (lab == c).any():
                C[c] = torch.nn.functional.normalize(X[lab == c].mean(0), dim=0)
    return lab


def sub_of(k):
    """辞書（Sudachi）の細かい分類（比べるためだけに使う。分け方には使わない）。"""
    w = k.split(":", 1)[1]
    m = tok.tokenize(w)
    for x in m:
        p = x.part_of_speech()
        if p[0] in ("助詞", "助動詞"):
            return p[0] + ("-" + p[1] if p[1] != "*" else "")
    return big[k]


def purity(lab, gold):
    tot = 0
    for c in set(lab.tolist()):
        tot += collections.Counter(gold[i] for i in range(len(gold)) if int(lab[i]) == c).most_common(1)[0][1]
    return tot / len(gold)


Xf = torch.stack([diff[k].flatten() for k in fw]); gold_f = [sub_of(k) for k in fw]
lab_f = kmeans(Xf, 8); res = dict(exp="91", function_words=len(fw), verbs=len(vb))
res["purity_function"] = purity(lab_f, gold_f)
res["purity_function_random"] = purity(torch.randint(0, 8, (len(fw),)), gold_f)
Xr = torch.stack([prof[k].flatten() for k in fw]); res["purity_function_raw"] = purity(kmeans(Xr, 8), gold_f)   # 差分を取らない（割合そのまま）で分けたとき
# 動詞: 直前の助詞の語（を・が・に・で・と）の割合 = 動詞が求める格（差分は動詞全体の平均から）
part_before = collections.defaultdict(collections.Counter)
for r in rows:
    for t, (w, l) in enumerate(r):
        if l.startswith("動詞-一般") and t > 0 and r[t - 1][1].startswith("助詞:"):
            part_before[f"動詞:{w}"][r[t - 1][1].split(":")[1]] += 1
CASES = ["を", "が", "に", "で", "と", "へ", "から", "は", "も", "て"]
V = torch.tensor([[part_before[k][c] for c in CASES] for k in vb], dtype=torch.float)
V = V / V.sum(1, keepdim=True).clamp(min=1); Vd = V - V.mean(0)
lab_v = kmeans(torch.cat([Vd, torch.stack([diff[k].flatten() for k in vb])], 1), 6)
L = ["# 品詞が何を求められているか（実験91）", "", "持ち主の案（2026-11-02）。作り方: experiments/exp91_function_word_demands.py。ラベルを使わず、前後の品詞の割合の差分（その語 − 同じ品詞の平均）だけで分けた。", "",
     f"## 助詞・助動詞（よく出る 60 語）", "",
     f"差分だけで 8 の仲間に分けると、辞書の細かい分類（格助詞・係助詞・副助詞・接続助詞・終助詞・準体助詞・助動詞）との一致（純度）: **{res['purity_function']:.2f}**"
     f"（差分を取らずに割合そのままで分けると {res['purity_function_raw']:.2f}、でたらめ {res['purity_function_random']:.2f}）。", "",
     "| 仲間 | 語（辞書の分類） | 求めるもの（差分の大きい所） |", "|---|---|---|"]
for c in sorted(set(lab_f.tolist())):
    ks = [k for i, k in enumerate(fw) if int(lab_f[i]) == c]
    L.append(f"| {c} | {'・'.join(f'{k.split(chr(58), 1)[1]}（{sub_of(k).split(chr(45))[-1]}）' for k in ks[:10])} | {demands(ks[0])} |")
L += ["", "## 語ごとの「求めるもの」（よく出る 20 語）", ""] + [f"- **{k.split(':', 1)[1]}**（{sub_of(k)}）: {demands(k)}" for k in fw[:20]]
L += ["", "## 動詞（よく出る 200 語）: 直前の助詞の割合の差分で 6 の仲間に分ける（求める格 = 自動・他動など）", "", "| 仲間 | 動詞（よく出る順） | 直前の助詞で多いもの（全体の平均との差） |", "|---|---|---|"]
for c in sorted(set(lab_v.tolist())):
    ids = [i for i in range(len(vb)) if int(lab_v[i]) == c]
    md = Vd[ids].mean(0); top = torch.argsort(md, descending=True)[:3]
    L.append(f"| {c} | {'・'.join(vb[i].split(':')[1] for i in ids[:14])} | {'・'.join(f'{CASES[int(j)]} +{float(md[j]):.2f}' for j in top)} |")
res["lines"] = L
with open("results/exp91.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
open("docs/function_word_demands_91.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
