"""実験103: 構造と時の順を、ホログラフィックに 1 枚の光に畳む（持ち主の問い 2026-11-02「構造や時系列をホログラフィックな処理で行う手段のアイディアがない」）

考え（結びつけ = 掛け算、重ね合わせ = 足し算、取り出し = 逆の位相を掛ける）:
  語・役割・時刻を、長さ D の「位相だけの光」（各点の明るさ 1、位相がでたらめ）で表す。光では位相の板そのもの。
  結びつけ: 役割 ⊙ 中身（点ごとの掛け算 = 位相の足し算 = 板を通すこと）。例: 主語 ⊙ 太郎。
  出来事: 1 つの述語の文節と、それに係る節の 役割 ⊙ 中身 を足し合わせた光。e = 述語 ⊙ 見つける + 主語 ⊙ 太郎 + 目的語 ⊙ 亀 + 場所 ⊙ 浜辺。
  時の順: 時刻 k の鍵 τ^k（1 歩ごとに位相が決まった量だけ回る光）を掛けて足す。物語 = Σ_k τ^k ⊙ e_k。1 枚の光（長さ D）に 20 文が全部入る。
  取り出し: 物語 ⊙ τ^{-k}（逆の位相）で k 番目の出来事が浮かぶ（ほかの出来事はばらばらの雑音になる）。さらに ⊙ 役割^{-1} で、その役割の中身が浮かぶ。
           浮かんだ光を、語の光の一覧（名札）と重ねて、いちばん近い語を答えにする（掃除）。
  質問: 手がかり（主語 ⊙ 太郎 + 述語 ⊙ 買い取る …）と、各時刻の出来事の光の重なりで時刻 k を選び、答えの役割で取り出す。
  次・前: 時刻 k が分かれば、τ^{k+1}（位相を 1 歩回す）で次の出来事を取り出せる。
語の光: 語ごとにでたらめな位相（名札）+ 語板 v4 の光から作った位相（意味の近い語は少し似る、重み 0.3）。役割: が・は → 主語、を → 目的語、に・へ → 行き先、で → 場所・手段、
       という → 名前、どんな → 様子、の → 持ち主、引用 → 言った中身、述語、ほか（実験87 の役割を読み替えた小さな表）。中身 = 節の主な語（名詞・動詞・形容詞）。
比べ: 実験97（文章樹 + 述語 2 倍）29/30。D = 256・1024・4096 で、取り出しの雑音を見る。
結果: results/exp103.txt、docs/holographic_story_103.md、docs/figs/holographic_story_103.svg。実行: python experiments/exp103_holographic_story.py
"""
import json, math, os, sys

import torch

spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; sys.argv = [sys.argv[0]]; exec(compile(src, spec_path, "exec"), ns)
T, S, V4, Q1, A1, Q2, A2, WH, CONTENT = (ns[k] for k in ("T", "S", "V4", "Q1", "A1", "Q2", "A2", "WH", "CONTENT"))
same = ns["build"].__globals__["same"]; subtree_text = ns["subtree_text"]
sys.stdout.reconfigure(encoding="utf-8")
ROLE = {"が": "主語", "は": "主語", "も": "主語", "を": "目的語", "に": "行き先", "へ": "行き先", "で": "場所", "から": "起点", "と": "相手", "という": "名前",
        "どんな": "様子", "の": "持ち主", "引用": "言った中身"}
ROLES = sorted(set(ROLE.values())) + ["述語", "ほか", "係り先", "袋"]
WH_ROLE = {"何を": ["目的語", "ほか"], "何に": ["行き先"], "何が": ["主語"], "何と": ["言った中身"], "どこへ": ["行き先"], "どこで": ["場所"], "誰": ["主語", "行き先"],
           "どんな": ["様子"], "名前": ["名前"], "何": ["目的語", "主語", "行き先"]}


def main_word(n):
    ws = [w for w in n["words"] if w not in ("為る", "居る", "成る", "おる", "様")]
    return ws[-1] if n["role"] in ("どんな",) and ws else (ws[0] if ws else None)


def word_of(n):
    """枠に入れる語: 名前の節は語をつなげる（浦島+太郎）、ほかは主な語。"""
    if n["role"] == "という" and n["words"]:
        return "".join(w for w in n["words"] if w not in ("言う", "と"))
    return main_word(n)


BAG = 1.0                                                                                      # 袋（役割なしの語の束）の重み


def run(D, mode, seed=0):
    g = torch.Generator().manual_seed(seed)
    rand_phase = lambda: torch.exp(1j * 2 * math.pi * torch.rand(D, generator=g))
    vocab = sorted({w for nodes in T for n in nodes for w in n["words"]} | {word_of(n) for nodes in T for n in nodes if word_of(n)} | {w[1] for q in Q1 + Q2 for b in q["bunsetsu"] for w in b["words"]})
    proj = torch.randn(next(iter(V4.values())).shape[0], D, generator=g) / 30                 # 語板 v4 の光 → 位相（意味の近い語は位相が少し似る）
    W = {}
    for w in vocab:
        ph = 2 * math.pi * torch.rand(D, generator=g)
        if w in V4:
            ph = ph + 0.3 * (V4[w] @ proj) * 2 * math.pi
        W[w] = torch.exp(1j * ph)
    R = {r: rand_phase() for r in ROLES}
    step = torch.exp(1j * 2 * math.pi * torch.rand(D, generator=g))                           # 時刻の 1 歩（位相を回す量）
    tau = lambda k: step ** k
    sub = torch.exp(1j * 2 * math.pi * torch.rand(D, generator=g))                            # 文の中の 1 歩
    # 出来事の作り方:
    #   "文": 1 文 = 1 出来事（根の述語 + 根に係る節）。1 回目。
    #   "枠": 子を持つ節ごとに 1 つの枠（中心 = その節の語、まわり = 子の節を役割で結ぶ、親の節の語 = 「係り先」）。1 文に複数の枠。
    #         どの枠にも「袋」（役割なしの語の束 = 袋 ⊙ 語）を足す。質問の時刻選びは 役割つき + 袋 の両方で重ねる。
    events = []
    for si, nodes in enumerate(T):
        if mode == "文":
            root = next(n for n in nodes if n["head"] == -1); e = torch.zeros(D, dtype=torch.complex64); items = []
            pw = main_word(root)
            if pw:
                e += R["述語"] * W[pw]; items.append(("述語", pw, root["text"]))
            for n in nodes:
                if n is root:
                    continue
                mw = main_word(n)
                if not mw:
                    continue
                role = ROLE.get(n["role"], "ほか")
                if n["head"] == root["id"] or role in ("様子", "名前", "持ち主", "言った中身"):
                    e += R[role] * W[mw]; items.append((role, mw, n["text"]))
            events.append((e, items, si))
            continue
        byid = {n["id"]: n for n in nodes}
        for c in nodes:
            kids = [n for n in nodes if n["head"] == c["id"] and word_of(n)]
            if not kids or not word_of(c):
                continue
            e = torch.zeros(D, dtype=torch.complex64); items = [("述語", word_of(c), c["text"])]
            if c["head"] in byid and word_of(byid[c["head"]]):
                items.append(("係り先", word_of(byid[c["head"]]), byid[c["head"]]["text"]))
            items += [(ROLE.get(n["role"], "ほか"), word_of(n), n["text"]) for n in kids]
            for r, w, _ in items:
                e += R[r] * W[w] + BAG * R["袋"] * W[w]
            events.append((e, items, si))
    # 時刻の鍵: 文の番号 k → τ^k、文の中の枠の番号 j → σ^j（文の中でも順番がある）
    keys, cnt = [], {}
    for _, _, si in events:
        j = cnt.get(si, 0); cnt[si] = j + 1; keys.append(tau(si) * sub ** j)
    story = sum(keys[k] * e for k, (e, _, _) in enumerate(events))                               # 1 枚の光に 20 文
    unit = lambda v: v / v.abs().pow(2).sum().sqrt().clamp(min=1e-9)
    sim = lambda a, b: float((unit(a).conj() * unit(b)).sum().real)

    def cleanup(v, cands):
        return max(cands, key=lambda w: sim(v, W[w]))

    def ask(q):
        txt = q["sentence"]; wh = next((k for k in ("名前", "何を", "何に", "何が", "何と", "どこへ", "どこで", "誰", "どんな", "何") if k in txt), "何")
        stop = {"何", "誰", "どこ", "名前", "どんな", "最後"}; probe = torch.zeros(D, dtype=torch.complex64)
        for b in q["bunsetsu"]:
            for w in b["words"]:
                if w[2].startswith(CONTENT) and w[1] not in stop and w[0] not in stop and w[1] in W:
                    v = w[2].startswith("動詞") or (w[2].startswith("名詞") and any(x[1] == "為る" for x in b["words"]))
                    if v and w[1] in ("為る", "居る", "成る"):
                        continue
                    role = "述語" if v else ROLE.get(ns["build"].__globals__["role"](dict(words=[x[:3] for x in b["words"]], text=b["text"], dep="x")), "ほか")
                    probe += (2.0 if v else 1.0) * (R[role] + BAG * R["袋"]) * W[w[1]]                         # 手がかり: 役割 ⊙ 語（述語 2 倍、実験97）
        ks = sorted(range(len(events)), key=lambda k: -sim(story * keys[k].conj(), probe))    # 時刻ごとに取り出して手がかりと重ねる
        k = ks[0]; ev = story * keys[k].conj()
        cands = sorted({it[1] for it in events[k][1]} - {w[1] for b in q["bunsetsu"] for w in b["words"]})
        if not cands:
            return "（なし）", events[k][2]
        text = {it[1]: it[2] for it in events[k][1] if len(it) > 2}
        for r in WH_ROLE[wh] + ["係り先"]:                                                   # 役割の順に取り出す（目的語 → ほか → 係り先）
            sc, w = max((sim(ev * R[r].conj(), W[w]), w) for w in cands)
            if sc > 0.025:
                return text.get(w, w), events[k][2]
        return text.get(w, w), events[k][2]

    out = []
    for Q, A in ((Q1, A1), (Q2, A2)):
        for q, ans in zip(Q, A):
            a, k = ask(q); ok = any(a in x or x in a for x in ans) if a != "（なし）" else False
            out.append(dict(q=q["sentence"], ans=a, sent=k + 1, ok=ok))
    # 次の出来事: 時刻 k の次（τ^{k+1}）の述語を取り出す
    nxt = []
    first = {}
    for k, (_, _, si) in enumerate(events):
        first.setdefault(si, k)
    verbs = sorted({it[1] for e in events for it in e[1] if it[0] == "述語"})
    for si in sorted(first)[:-1]:
        if si + 1 not in first:
            continue
        ev = story * tau(si + 1).conj() * R["述語"].conj()                                    # 「次」 = 位相を 1 歩回すだけ
        got = cleanup(ev, verbs); want = events[first[si + 1]][1][0][1]
        nxt.append(got == want)
    # 取り出しの雑音: 正しい語との重なりと、ほかの語との重なり
    sig, noi, acc = [], [], []
    for k, (e, items, _) in enumerate(events):
        ev = story * keys[k].conj()
        for it in items:
            v = ev * R[it[0]].conj(); sig.append(sim(v, W[it[1]]))
            noi.append(max(sim(v, W[w]) for w in list(W)[:200] if w != it[1]))
            acc.append(cleanup(v, list(W)) == it[1])                                           # 時刻と役割が分かっているとき、全語（約 500）から正しく取り出せるか
    return dict(D=D, mode=mode, n_events=len(events), q20=sum(r["ok"] for r in out[:20]), q10=sum(r["ok"] for r in out[20:]), next_ok=sum(nxt), next_n=len(nxt),
                signal=sum(sig) / len(sig), recall=sum(acc) / len(acc), n_items=len(acc), noise=sum(noi) / len(noi), rows=out, events=[[si + 1] + [list(x[:2]) for x in it] for _, it, si in events])


res = dict(exp="103", runs=[run(D, m) for m in ("文", "枠") for D in (256, 1024, 4096, 16384)])
for r in res["runs"]:
    print(r["mode"], "出来事", r["n_events"], "D", r["D"], "質問", r["q20"], "+", r["q10"], "次の出来事", r["next_ok"], "/", r["next_n"], "取り出し正答", round(r["recall"], 3), r["n_items"], "重なり 正しい語", round(r["signal"], 3), "ほかの語の最大", round(r["noise"], 3), flush=True)
with open("results/exp103.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
best = res["runs"][-1]
L = ["# 構造と時の順をホログラフィックに 1 枚の光に畳む（実験103、浦島太郎）", "", "作り方: experiments/exp103_holographic_story.py。図: [figs/holographic_story_103.svg](figs/holographic_story_103.svg)", "",
     "| 出来事の作り方 | 光の長さ D | 質問 20 | 使っていない 10 | 次の出来事（19） | 時刻と役割が分かったときの取り出し正答 | 取り出しの重なり（正しい語 / ほかの語の最大） |", "|---|---|---|---|---|---|---|"]
for r in res["runs"]:
    L.append(f"| {r['mode']}（{r['n_events']} 個） | {r['D']} | {r['q20']} | {r['q10']} | {r['next_ok']} | {r['recall']:.0%}（{r['n_items']} 語） | {r['signal']:.3f} / {r['noise']:.3f} |")
L += ["", f"## 枠の光の中身（D = {best['D']}、述語ごと）", ""] + [f"- {it[0]} 文目: " + "、".join(f"{r} ⊙ {w}" for r, w in it[1:]) for it in best["events"]]
L += ["", f"## 質問の答え（D = {best['D']}）", "", "| 質問 | 答え | 時刻（文） | |", "|---|---|---|---|"] + [f"| {r['q']} | {r['ans']} | {r['sent']} | {'○' if r['ok'] else '×'} |" for r in best["rows"]]
open("docs/holographic_story_103.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
