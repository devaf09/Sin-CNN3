"""実験76: 前後からの属性変位 — 前の文 B と後ろの文 D から、間の文 C を作る属性と語板を集める（持ち主の構想 2026-10-30）

持ち主: 「後ろの文章に含まれる属性を光の融合で集め、確信度が高い属性を取り出す（試験では 3 つ）。前の文章にも同じように 3 つ。
         属性樹を 3 つの属性が共通する枝まで逆登りして、その途中の属性を順に拾う。共通する属性枝まで戻ったら、今度は前の文章の共通する枝まで登りながら途中の属性を拾う。
         試験では、前後の単語 1 つずつから、確信 30% 以上の上位 1 腕先の語板を 4 つずつ抽出し、属性変位で取得した属性の列の属性に近いものから 1 つずつ並べて語板の列を作る」。
      「後方への属性変位: 属性樹の枝の属性 C から B を引いて、その差分を B に付帯する（試験では C と B から）」。
言葉:
  1 腕先の語板 = 語板の属性の値が、それ自体語板を持つ語（「の前 X」「の後 Y」「と Z」の X・Y・Z）。結びつき = その値の明るさの割合（確信度）。
  属性 = (観点, 値)。確信度の高い属性 = 融合した明るさ × log(明るさ ÷ 全体の割合)（どの語にも多い「に 成る」ばかりにならないよう、珍しさを掛ける）。
方法（語板 v2 = 実験73、名詞 3 万語・観点 9。属性樹は実験71 と同じ作り方）:
  問題: 青空文庫 60 作品（実験70 と同じ選び方）の、続いた 3 文 B・C・D（どれも語板のある名詞を 2 つ以上含む）。C を隠し、B と D から C を当てる。
  1. 属性の流れ（前後からの属性変位）: D の名詞の語板を明るさで融合 → 確信度の高い属性 3 つ → 各属性で終わる節（D の名詞を最も多く含む）→ 3 つの節の共通の祖先（共通の枝）まで登りながら属性を拾う。
     B も同じ。D の共通の枝から、B の共通の枝まで（両方の共通の祖先を通って）行きながら属性を拾う。流れ = D の葉 → D の枝 → … → B の枝 → B の葉。
  2. 属性変位の語板の取得: B・D の名詞それぞれ 1 つ（その文でいちばん珍しい = 意味の濃い名詞）から、1 腕先の語板を 4 つずつ（確信度 30% 以上。足りなければ上位から）。
     流れの属性の順に、その属性がいちばん明るい語板を 1 つずつ取って並べる（語板の列）。
  当たりの測り方:
    (i) 流れの属性は C に合うか: C の名詞を融合した明るさが、流れの属性で、全体の割合の何倍か（持ち上がり）。比べ: B と D の上位属性だけ（流れなし）、でたらめの属性。
    (ii) C の名詞を、全部の名詞 3 万語から当てる（順位、MRR・上位 100 に入る割合）:
         流れの属性との重なり / B と D の融合との重なり（流れなし）/ 両方の順位の和 / 語の多さだけ。
    (iii) 語板の列（8 枚）のどれかが C に出る割合。比べ: 1 腕先の候補 8 枚を全部（並べ方なし = 同じ）、語の多さの上位 8、でたらめ 8。
  3. 後方への属性変位: 属性の差分（C の融合 − B の融合）を B に足したもの = C の像。差分を本当の C から作れば当たって当然なので、
     「D − B の差分の半分を B に足す」（間の文は、前と後ろの途中）が C の名詞をどれだけ当てるかを、B だけ・D だけ・B と D の平均と比べる。
結果: results/exp76.txt、例: docs/attribute_shift_76.md
実行: CUDA_VISIBLE_DEVICES= python experiments/exp76_attribute_shift.py
"""
import collections, json, math, os, pickle, random, re, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t00 = time.time()
PRIV = "D:/AI_git/Sin-CNN3_private"
# ---- 語板 v2（2 つに分けて学んだものを合わせる） ----
Ps = []; WORDS = []
for k in (0, 1):
    M = torch.load(f"{PRIV}/models/goban_v2_part{k}.pt", weights_only=False); Ps.append(M["P"]); WORDS += M["words"]
VIEWS = M["views"]; NVIEW, NVAL = 9, 45; WI = {w: i for i, w in enumerate(WORDS)}
INV = {(j, r): x for j, v in enumerate(VIEWS) for x, r in M["pos"][v].items()}            # 出口の場所 → 値の名前
with torch.no_grad():
    P = torch.cat(Ps); B3 = torch.cat([run(P[i:i + 5000], NVIEW, NVAL + NVIEW) for i in range(0, len(WORDS), 5000)])
B3 = B3 / B3.sum(-1, keepdim=True).clamp(min=1e-12)                                         # (語, 観点, 出口) 観点ごとの割合
A3 = B3[:, :, :NVAL]                                                                        # 属性 = 値の出口だけ（「その他」は中身が分からないので除く）
FL = A3.reshape(len(WORDS), -1)                                                             # 1 語 = 9 × 45 = 405 個の属性の明るさ
prior = FL.mean(0)                                                                          # 全体の割合（どの語にも多い属性を見分けるため）
FREQ = pickle.load(open(f"{PRIV}/data/views/merged2.pkl", "rb"))["noun_total"]              # 語の多さ
logf = torch.tensor([math.log(2 + FREQ.get(w, 0)) for w in WORDS])
aname = lambda a: f"{VIEWS[a // NVAL]} {INV.get((a // NVAL, a % NVAL), '?')}"              # 属性の番号 → 「観点 値」
print("語板", len(WORDS), f"{time.time() - t00:.0f} 秒", flush=True)

# ---- 属性樹（実験71 と同じ: 共通の度合いの大きい属性で枝分かれ、深さ 4、1 つの節から最大 6 本） ----
has = FL >= torch.clamp(3 * prior, min=0.05)                                               # (語, 属性) その属性を持つか
nodes = [dict(depth=0, members=torch.arange(len(WORDS)), attr=None, parent=None)]


def grow(i, used):
    nd = nodes[i]
    if nd["depth"] >= 4 or len(nd["members"]) < 20:
        return
    share = has[nd["members"]].float().mean(0)
    for u in used:
        share[u] = -1
    for a in torch.argsort(share, descending=True)[:6].tolist():
        if share[a] < 0.05:
            break
        mem = nd["members"][has[nd["members"], a]]
        if len(mem) < 10 or len(mem) == len(nd["members"]):
            continue
        nodes.append(dict(depth=nd["depth"] + 1, members=mem, attr=a, parent=i)); grow(len(nodes) - 1, used + [a])


grow(0, [])
memsets = [set(nd["members"].tolist()) for nd in nodes]
by_attr = collections.defaultdict(list)                                                     # 属性 → その属性で終わる節
for j, nd in enumerate(nodes):
    if nd["attr"] is not None:
        by_attr[nd["attr"]].append(j)
print("属性樹の節", len(nodes), f"{time.time() - t00:.0f} 秒", flush=True)


def up(j):
    """節 j から根までの節の並び（自分を含む）。"""
    out = []
    while j is not None:
        out.append(j); j = nodes[j]["parent"]
    return out


def lca(js):
    """節の並び js の共通の祖先（いちばん深いもの）= 共通の枝。"""
    common = set(up(js[0]))
    for j in js[1:]:
        common &= set(up(j))
    return max(common, key=lambda j: nodes[j]["depth"])


def top_attrs(ids, k=3):
    """名詞 ids の語板を明るさで融合し、確信度の高い属性 k 個（明るさ × log(明るさ ÷ 全体の割合)）。"""
    F = FL[ids].mean(0)
    return torch.argsort(F * torch.log((F + 1e-9) / (prior + 1e-9)), descending=True)[:k].tolist(), F


def side(ids):
    """1 つの文の側: 上位 3 属性 → 各属性で終わる節（その文の名詞を最も多く含む）→ 共通の枝。返り値: (葉の節の並び, 共通の枝, 上位属性)。"""
    ats, _ = top_attrs(ids); leaves = []
    for a in ats:
        if by_attr[a]:
            leaves.append(max(by_attr[a], key=lambda j: (len(memsets[j] & set(ids)), nodes[j]["depth"])))
    return leaves, (lca(leaves) if leaves else 0), ats


def flow(idsD, idsB):
    """前後からの属性変位: D の葉 → D の共通の枝 → (B との共通の祖先) → B の共通の枝 → B の葉、の順に、通った節の属性を拾う。"""
    lD, cD, aD = side(idsD); lB, cB, aB = side(idsB)
    seq = []

    def add(js):
        for j in js:
            a = nodes[j]["attr"]
            if a is not None and a not in seq:
                seq.append(a)
    for j in lD:                                                                            # D の葉から D の共通の枝まで登る
        add(up(j)[:up(j).index(cD)] if cD in up(j) else up(j))
    top = lca([cD, cB])
    add(up(cD)[:up(cD).index(top) + 1])                                                     # D の枝から両方の共通の祖先まで登る
    add(list(reversed(up(cB)[:up(cB).index(top)])))                                        # 共通の祖先から B の枝まで（下りながら）
    for j in lB:                                                                            # B の枝から B の葉まで
        path = up(j); add(list(reversed(path[:path.index(cB)] if cB in path else path)))
    for a in aD + aB:                                                                       # 樹に節がない上位属性も、最後に足す
        if a not in seq:
            seq.append(a)
    return seq, dict(D=aD, B=aB, cD=cD, cB=cB, top=top)


NOUNV = [VIEWS.index(v) for v in ("の前", "の後", "と")]                                     # 値が名詞の観点（1 腕先の語板になる）


def one_arm(w, k=4, th=0.30):
    """語 w の 1 腕先の語板: 「の前・の後・と」の値で語板を持つ語。確信度（その観点での割合）th 以上の上位 k 個（足りなければ上位から埋める）。"""
    c = []
    for v in NOUNV:
        for x in range(NVAL):
            y = INV.get((v, x))
            if y in WI and y != w:
                c.append((float(B3[WI[w], v, x]), y))
    c.sort(reverse=True); strong = [y for s, y in c if s >= th]
    out = []
    for y in strong + [y for _, y in c]:
        if y not in out:
            out.append(y)
    return out[:k], len(strong)


def one_arm_rel(w, k=4, ksd=3.0):
    """相対の確信度の 1 腕先: 観点ごとに、明るさが その観点の値の平均 + ksd × 標準偏差 を超える値（語板を持つ語）の上位 k 個。"""
    c = []
    for v in NOUNV:
        b = A3[WI[w], v]; th = float(b.mean() + ksd * b.std())
        for x in range(NVAL):
            y = INV.get((v, x))
            if y in WI and y != w and float(b[x]) > th:
                c.append((float(b[x]), y))
    return [y for _, y in sorted(c, reverse=True)][:k]


# ---- 問題: 青空文庫の続いた 3 文 ----
from sudachipy import Dictionary, SplitMode
tok = Dictionary(dict="core").tokenizer(); MODE = SplitMode.C
docs = [json.loads(l) for l in open(f"{PRIV}/data/text/aozora_full.jsonl", encoding="utf-8")]
rnd = random.Random(0); docs = [d for d in docs if 30000 < len(d["text"]) < 300000]; rnd.shuffle(docs); docs = docs[:60]
nouns = lambda s: sorted({m.normalized_form() for m in tok.tokenize(s[:500], MODE) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] == "普通名詞" and m.normalized_form() in WI})
stats = collections.defaultdict(float); n_q = 0; ex = []; LIST = []
rk = lambda x: torch.argsort(torch.argsort(-x)).float()                                     # 大きい順の順位（0 から）
g = torch.Generator().manual_seed(0)
for d in docs:
    sents = [s for s in re.split(r"(?<=。)", d["text"].replace("\r", "").replace("\n", "").replace("　", "")) if len(s) > 8 and "---" not in s and "底本" not in s and "テキスト中" not in s]   # 作品の頭・終わりの説明は除く
    NS = [nouns(s) for s in sents]
    picked = 0
    for k in range(1, len(sents) - 1, 7):                                                   # 7 文おきに、続いた 3 文
        nB, nC, nD = NS[k - 1], NS[k], NS[k + 1]
        if len(nB) < 2 or len(nC) < 2 or len(nD) < 2 or picked >= 25:
            continue
        cset = set(nC) - set(nB) - set(nD)                                                 # B・D に出ない C の名詞を当てる（同じ語を探すだけにならないように）
        if not cset:
            continue
        picked += 1; n_q += 1
        iB = [WI[w] for w in nB]; iD = [WI[w] for w in nD]; iC = [WI[w] for w in cset]
        seq, info = flow(iD, iB)
        # (i) 流れの属性が C に合うか（C の融合の、全体の割合に対する持ち上がり）
        FC = FL[iC].mean(0); lift = lambda ats: float((FC[ats] / prior[ats]).mean()) if ats else 0.0
        stats["持ち上がり: 流れの属性"] += lift(seq); stats["持ち上がり: B と D の上位属性だけ"] += lift(info["D"] + info["B"])
        stats["持ち上がり: でたらめの属性"] += lift(torch.randint(0, FL.shape[1], (len(seq),), generator=g).tolist())
        stats["流れの属性の数"] += len(seq)
        # (ii) C の名詞を 3 万語から当てる
        w_seq = torch.zeros(FL.shape[1]); w_seq[seq] = 1.0
        s_flow = (FL / (prior + 1e-9)) @ w_seq                                              # 流れの属性の持ち上がりの和
        FBD = FL[iB + iD].mean(0); s_fuse = torch.nn.functional.cosine_similarity(FL, FBD[None], dim=1)   # B と D の融合との重なり（流れなし）
        sc = {"流れの属性": s_flow * logf, "B と D の融合（流れなし）": s_fuse * logf, "語の多さだけ": logf.clone()}
        sc["流れ + 融合（順位の和）"] = -(rk(sc["流れの属性"]) + rk(sc["B と D の融合（流れなし）"]))
        # 語の多さ（P(語)）を、掛け算でなく順位の和で合わせる（1 回目の試しで、明るさの重なり × log(多さ) は「多さだけ」に負けたため）
        sc["多さ + 融合（順位の和）"] = -(rk(logf) + rk(s_fuse))
        sc["多さ + 流れ（順位の和）"] = -(rk(logf) + rk(s_flow))
        sc["多さ + 融合 + 流れ（順位の和）"] = -(rk(logf) + rk(s_fuse) + rk(s_flow))
        # ベイズの形: log P(語) + β × (重なりを標準化したもの)。β で「文脈をどれだけ信じるか」を変える
        z = lambda x: (x - x.mean()) / (x.std() + 1e-9); lpw = torch.log(torch.exp(logf) - 2 + 1)   # log(多さ + 1)
        for beta in (0.5, 1.0, 2.0):
            sc[f"log 多さ + {beta} × 融合"] = lpw + beta * z(s_fuse)
            sc[f"log 多さ + {beta} × 流れ"] = lpw + beta * z(s_flow)
            sc[f"log 多さ + {beta} × (融合 + 流れ)"] = lpw + beta * z(z(s_fuse) + z(s_flow))
        # 3. 後方への属性変位: B に (D − B) の半分を足した像（前と後ろの途中）
        sc["B + (D − B)/2（属性変位の像）"] = torch.nn.functional.cosine_similarity(FL, (FL[iB].mean(0) + 0.5 * (FL[iD].mean(0) - FL[iB].mean(0)))[None], dim=1) * logf
        sc["B だけ"] = torch.nn.functional.cosine_similarity(FL, FL[iB].mean(0)[None], dim=1) * logf
        sc["D だけ"] = torch.nn.functional.cosine_similarity(FL, FL[iD].mean(0)[None], dim=1) * logf
        for name, s in sc.items():
            s = s.clone(); s[iB + iD] = -1e9                                                 # B・D に出た語は除く
            r = rk(s)[iC]; best = float(r.min())
            stats[f"MRR: {name}"] += 1 / (best + 1); stats[f"上位100: {name}"] += float(best < 100)
        # (iii) 属性変位の語板の取得: B・D のいちばん珍しい名詞から 1 腕先 4 つずつ → 流れの順に並べる
        wB = min(nB, key=lambda w: FREQ.get(w, 0)); wD = min(nD, key=lambda w: FREQ.get(w, 0))
        cB, sB = one_arm(wB); cD, sD = one_arm(wD); pool = [y for y in cD + cB if y not in nB and y not in nD]
        stats["確信度 30% 以上の 1 腕先（1 語あたり）"] += (sB + sD) / 2
        order = []
        for a in seq:                                                                       # 流れの属性の順に、その属性がいちばん明るい語板を 1 つずつ
            rest = [y for y in pool if y not in order]
            if not rest:
                break
            order.append(max(rest, key=lambda y: float(FL[WI[y], a])))
        order += [y for y in pool if y not in order]
        stats["語板の列のどれかが C に出る"] += float(bool(set(order) & cset))
        # 確信度を「割合 30%」でなく、相対（その観点の値の平均 + 3 × 標準偏差 を超える = 実験72 の過飽和の式と同じ考え）で選んだ 1 腕先
        rel = lambda w: one_arm_rel(w)
        pr = [y for y in rel(wD) + rel(wB) if y not in nB and y not in nD]
        stats["1 腕先（相対の確信度、平均 + 3σ）の数"] += len(pr) / 2
        stats["1 腕先（相対）のどれかが C に出る"] += float(bool(set(pr[:8]) & cset))
        stats["語板の列の 1 番目が C に出る"] += float(bool(order) and order[0] in cset)
        top8 = [WORDS[int(j)] for j in torch.argsort(logf, descending=True) if WORDS[int(j)] not in nB + nD][:8]
        stats["語の多さの上位 8 が C に出る"] += float(bool(set(top8) & cset))
        rnd8 = [WORDS[int(j)] for j in torch.randint(0, len(WORDS), (8,), generator=g)]
        stats["でたらめ 8 が C に出る"] += float(bool(set(rnd8) & cset))
        # ---- 一覧用（持ち主の依頼 2026-10-30）: 流れに対応した語板と、語板どうしのつながりで選んだ語板 ----
        flow_boards = []                                                                    # 流れの属性ごとに、その属性がいちばん持ち上がる語板（よく出る語を少し優先）
        for a in seq:
            sa = FL[:, a] / (prior[a] + 1e-9) * logf; sa[iB + iD] = -1
            for j in torch.argsort(sa, descending=True)[:5].tolist():
                if WORDS[j] not in flow_boards:
                    flow_boards.append(WORDS[j]); break
        links = {}                                                                          # つながり: B・D の名詞と流れの語板の 1 腕先（相対の確信度）
        for w in nB + nD + flow_boards:
            for y in one_arm_rel(w, k=6):
                if y not in nB and y not in nD:
                    links.setdefault(y, []).append(w)
        fb = set(flow_boards)
        link_boards = sorted(links, key=lambda y: (-(y in fb), -len(links[y]), -float(s_flow[WI[y]])))[:8]   # 流れの語板でもあるもの → つながりが多いもの → 流れに近いもの
        got = set(flow_boards) | set(link_boards); cn = [w for w in nC if w not in nB and w not in nD]
        rate = len(got & set(cn)) / len(cn)
        stats["一致率（流れの語板 + つながりの語板）"] += rate
        cut2 = lambda s: ((s[:80] + "…") if len(s) > 80 else s).replace("|", "｜")
        LIST.append(dict(rate=rate, title=d.get("title", ""), B=cut2(sents[k - 1]), C=cut2(sents[k]), D=cut2(sents[k + 1]), flow=" → ".join(aname(a) for a in seq[:6]),
                         fboards="・".join(flow_boards[:6]), lboards="・".join(f"{y}（←{'・'.join(links[y][:2])}）" for y in link_boards[:6]),
                         hit="・".join(sorted(got & set(cn))) or "−", cn="・".join(cn)))
        if len(ex) < 40 and picked <= 1:
            cut = lambda s: ((s[:60] + "…") if len(s) > 60 else s).replace("|", "｜")
            ex.append(dict(title=d.get("title", ""), B=cut(sents[k - 1]), C=cut(sents[k]), D=cut(sents[k + 1]), flow=" → ".join(aname(a) for a in seq[:8]),
                           arm=f"{wD}: {'・'.join(cD)} ／ {wB}: {'・'.join(cB)}", order="・".join(order), cset="・".join(sorted(cset))))
res = {k: v / n_q for k, v in stats.items()}; res["問題の数"] = n_q; res["属性樹の節"] = len(nodes); res["sec"] = time.time() - t00
with open("results/exp76.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(dict(exp="76", **res), ensure_ascii=False) + "\n")
lines = ["# 前後からの属性変位の例（実験76）", "", "持ち主の構想（2026-10-30）。前の文 B と後ろの文 D から、間の文 C を作る属性（属性の流れ）と語板の列を集めた。作り方: experiments/exp76_attribute_shift.py。",
         "1 作品から 1 例。「C の名詞」は B・D に出ない C の名詞（当てたいもの）。「1 腕先の語板」は、B・D のいちばん珍しい名詞の「の前・の後・と」の値で、語板を持つ語 4 つずつ。", "",
         "| 文 | 属性の流れ（D → B、先頭 8） | 1 腕先の語板（D の語: … ／ B の語: …） | 語板の列 | C の名詞 |", "|---|---|---|---|---|"]
for e in ex:
    lines.append(f"| B: {e['B']}<br>C: {e['C']}<br>D: {e['D']} | {e['flow']} | {e['arm']} | {e['order']} | {e['cset']} |")
lines += ["", "## 数（問題 " + str(n_q) + " 問）", ""] + [f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}" for k, v in res.items()]
open("docs/attribute_shift_76.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
# 一覧: 名詞の一致率が高い 100 例・低い 100 例（低い方は、一致 0 の中から間の行の名詞が多いもの）
LIST.sort(key=lambda e: (-e["rate"], -len(e["cn"])))
hi = LIST[:100]; lo = sorted(LIST[-100:] if len(LIST) >= 200 else LIST[100:], key=lambda e: e["rate"])
L = ["# 属性変位の一覧（実験76）— 名詞の一致率が高い 100 例・低い 100 例", "",
     "持ち主の依頼（2026-10-30）。青空文庫（著作権切れ）の小説などから続いた 3 行（前の行 B・間の行 C・後ろの行 D）を取り、B と D だけから C を作る属性と語板を集めた。作り方: experiments/exp76_attribute_shift.py。", "",
     "- **属性の流れ** = 後ろの行 D の上位 3 属性から属性樹を共通の枝まで逆に登り、前の行 B の共通の枝まで登りながら拾った属性（先頭 6）。",
     "- **流れに対応した語板** = 流れの属性ごとに、その属性がいちばん持ち上がる語板（よく出る語を少し優先、B・D の語は除く）。",
     "- **つながりで選んだ語板** = B・D の名詞と流れの語板の「1 腕先」（「の前・の後・と」の値で語板を持つ語、相対の確信度 = 平均 + 3σ 超え）。（← どの語板から来たか）",
     "- **一致率** = 間の行 C の名詞（B・D に出ないもの）のうち、流れの語板とつながりの語板のどちらかに入った割合。",
     f"- 全部 {len(LIST)} 例の平均の一致率: {res.get('一致率（流れの語板 + つながりの語板）', 0):.3f}。一致率が 0 より大きい例: {sum(e['rate'] > 0 for e in LIST)}。", ""]
for name, part in (("一致率が高い 100 例", hi), ("一致率が低い 100 例", lo)):
    L += [f"## {name}", "", "| # | 後ろの行 D | 属性の流れ | 流れに対応した語板 | つながりで選んだ語板 | 前の行 B | 間の行 C（答え） | 当たった名詞 / C の名詞 | 一致率 |", "|---|---|---|---|---|---|---|---|---|"]
    for i, e in enumerate(part, 1):
        L.append(f"| {i} | {e['D']} | {e['flow']} | {e['fboards']} | {e['lboards']} | {e['B']} | {e['C']} | {e['hit']} / {e['cn']} | {e['rate']:.2f} |")
    L.append("")
open("docs/attribute_shift_76_list.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
for k, v in res.items():
    print(k, round(v, 4) if isinstance(v, float) else v)
