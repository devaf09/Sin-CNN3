"""属性逆引き表（持ち主の依頼 2026-10-29）: 語板 A − 語板 B の「光の像」を、ほかの語板に逆から入れて反応する語を出し、属性樹の幹とその仲間を表示する。

持ち主: 「りんご − トマト = 出てきた光の像（色々含まれる）→ 1. 他の語板に逆から入れて反応する語を出す。2. 属性樹走査できたら、その幹の他の内容を表示する。
        適当な名詞や動詞をピックアップして 100 個ほど」。
作り方:
  光の像 = A の語板の出口の明るさ（観点ごとの割合）− B の語板の明るさ（学び 44: 明るさの引き算）。正の所 = A にあって B にない（少ない）使われ方。
  1. 逆から入れる: 光の像の正の所を、ほかの全部の語板の出口に逆から入れると、各語板の入口に戻る光の明るさ = 光の像とその語板の明るさの重なり
     （逆向きの明るさ = 順向きの明るさ、学び 21・実験68）。重なりの大きい語 = 反応した語（A・B 自身は除く）。重なりは cos で測る。
  2. 属性樹の幹: 実験71 の属性樹（名詞 2 万語）・同じ作り方の動詞の属性樹（3000 語）で、節の「属性差グレイン」（節の融合した板 − 親の融合した板）が
     光の像にいちばん重なる節を探し（属性樹の走査）、その道筋（幹）と、節の仲間の語を出す。
語板: 実験59 の名詞 2 万語・動詞 3000 語（非公開フォルダ）。
出力: docs/diff_reverse_table.md。実行: python tools/diff_reverse_table.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import run

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4)


def load(kind, nview):
    M = torch.load(f"D:/AI_git/Sin-CNN3_private/models/goban_{kind}_seed0.pt", weights_only=False)
    with torch.no_grad():
        B3 = torch.cat([run(M["P"][i:i + 5000], nview, 47 + nview) for i in range(0, len(M["words"]), 5000)])
    B3 = B3 / B3.sum(-1, keepdim=True).clamp(min=1e-12)                                     # 観点ごとに割合に直す
    views = list(M["vals"].keys())
    names = {}
    for j, v in enumerate(views):                                                            # 出口の場所 → 値の名前（実験58 のずらし）
        for r, x in enumerate(M["vals"][v]):
            names[(j, (r + 9 * j) % 47)] = f"{v} {x}"
    return M["words"], B3, views, names


def load_v2():
    """語板 v2（実験73、観点 9・値 45 + その他、名詞 3 万語。リフレッシュ側の 2 つの片方を合わせる）。"""
    P = []; words = []
    for k in (0, 1):
        M = torch.load(f"D:/AI_git/Sin-CNN3_private/models/goban_v2_part{k}.pt", weights_only=False)
        P.append(M["P"]); words += M["words"]
    P = torch.cat(P); views = M["views"]; nv = len(views)
    with torch.no_grad():
        B3 = torch.cat([run(P[i:i + 5000], nv, 45 + nv) for i in range(0, len(words), 5000)])
    B3 = B3 / B3.sum(-1, keepdim=True).clamp(min=1e-12)                                     # 観点ごとに割合に直す
    names = {(j, r): f"{v} {x}" for j, v in enumerate(views) for x, r in M["pos"][v].items()}   # 出口の場所 → 値の名前（v2 は pos に場所が入っている）
    return words, B3, views, names


def tree(B3):
    """実験71 と同じ属性樹（共通の度合いの大きい属性で枝分かれ、深さ 4、枝 最大 6）。節: 道筋・仲間・融合した板・属性差グレイン。"""
    W, V, X = B3.shape; NVAL = X - V; ATTR = [(v, x) for v in range(V) for x in range(NVAL)]   # 「その他」（後ろの V 個）は枝分かれに使わない
    prior = B3.mean(0); has = torch.stack([B3[:, v, x] >= max(0.05, 3 * float(prior[v, x])) for v, x in ATTR], 1)
    nodes = [dict(depth=0, members=torch.arange(W), path=[], parent=None)]

    def grow(i, used):
        nd = nodes[i]
        if nd["depth"] >= 4 or len(nd["members"]) < 20:
            return
        share = has[nd["members"]].float().mean(0)
        for a in used:
            share[a] = -1
        for a in torch.argsort(share, descending=True)[:6].tolist():
            if share[a] < 0.05:
                break
            mem = nd["members"][has[nd["members"], a]]
            if len(mem) < 10 or len(mem) == len(nd["members"]):
                continue
            nodes.append(dict(depth=nd["depth"] + 1, members=mem, path=nd["path"] + [ATTR[a]], parent=i)); grow(len(nodes) - 1, used + [a])
    grow(0, [])
    flat = B3.reshape(W, -1)
    for nd in nodes:
        nd["fused"] = flat[nd["members"]].mean(0)
    for nd in nodes:
        nd["diff"] = nd["fused"] - (nodes[nd["parent"]]["fused"] if nd["parent"] is not None else nd["fused"])   # 属性差グレイン
    return nodes


NOUN_PAIRS = """リンゴ トマト|犬 猫|電車 バス|先生 学生|海 山|夏 冬|朝 夜|ラーメン うどん|コーヒー 紅茶|本 映画|父 母|兄 弟|医者 看護師|学校 会社|車 自転車|雨 雪|空 海|火 水|
東京 大阪|手紙 電話|ビール 牛乳|パン ご飯|肉 魚|花 木|鳥 魚|王 兵|刀 銃|城 家|川 道|月 太陽|春 秋|子供 大人|男 女|友達 恋人|警察 犯人|病院 学校|
テレビ ラジオ|ゲーム 漫画|音楽 絵|歌 踊り|料理 掃除|仕事 遊び|夢 現実|心 体|頭 手|目 耳|声 顔|金 時間|戦争 平和|勝利 敗北|喜び 悲しみ|
愛 憎しみ|神 悪魔|魔法 剣|ドラゴン 狼|部屋 庭|窓 扉|机 椅子|服 靴|帽子 眼鏡|鍵 財布|薬 毒|星 雲|風 光|石 砂|紙 布|ケーキ 寿司|野菜 果物|旅行 引っ越し""".replace("\n", "")
VERB_PAIRS = """食べる 飲む|走る 歩く|見る 聞く|書く 読む|買う 売る|行く 来る|笑う 泣く|作る 壊す|開ける 閉める|始める 終わる|教える 習う|勝つ 負ける|
入る 出る|持つ 捨てる|生まれる 死ぬ|寝る 起きる|座る 立つ|歌う 踊る|話す 黙る|愛する 憎む|待つ 急ぐ|探す 見付ける|殴る 蹴る|洗う 拭く|切る 繋ぐ|
登る 降りる|投げる 受ける|貸す 借りる|呼ぶ 答える|守る 攻める""".replace("\n", "")
out = ["# 属性逆引き表 — 語板 A − 語板 B の「光の像」から、反応する語と属性樹の幹を引く", "",
       "持ち主の依頼（2026-10-29）。作り方は tools/diff_reverse_table.py。語板は実験59（名詞 2 万語・動詞 3000 語、非公開の材料から）。",
       "",
       "- **光の像** = A の語板の出口の明るさ − B の語板の明るさ（明るさの引き算）。正の所 = A にあって B に少ない使われ方。表には強い所を 3 つ。観点ごとの「その他」（上位 47 個以外をまとめた所）は中身が分からないので使わない。",
       "- **逆から入れて反応した語** = 光の像（正の所）を、ほかの全部の語板の出口に逆から入れたとき、入口に戻る光が明るい語（光の像との重なり × 語のよく出る度合いの対数、上位 5）。",
       "- **属性樹の幹** = 節の属性差グレイン（節 − 親）が光の像にいちばん重なる節の道筋。**幹の仲間** = その節の語（光の像との重なりが大きい順に 5）。",
       ""]
from sudachipy import Dictionary
_tok = Dictionary(dict="core").create()
_C = json.load(open("D:/AI_git/Sin-CNN3_private/data/views/counts_all.json", encoding="utf-8"))["train"]   # 語のよく出る度合い（逆引きの重み、学び 59・61）
FREQ = {"名詞": {n: sum(sum(c.values()) for c in vs.values()) for n, vs in _C.items()}, "動詞": {}}
for n, vs in _C.items():
    for v in ("を", "が", "で", "に"):
        for x, m in vs.get(v, {}).items():
            FREQ["動詞"][x] = FREQ["動詞"].get(x, 0) + m
del _C
norm = lambda w: _tok.tokenize(w)[0].normalized_form()                                     # 語板は正規形（林檎・饂飩）なので、表の語を正規形に直して探す
V2 = os.environ.get("V2") == "1"                                                           # V2=1: 語板 v2（実験73、名詞だけ）で作り直す
if V2:
    import pickle
    FREQ["名詞"] = pickle.load(open("D:/AI_git/Sin-CNN3_private/data/views/merged2.pkl", "rb"))["noun_total"]   # v2 の材料での語の多さ
    out[2] = "持ち主の依頼（2026-10-29）。作り方は tools/diff_reverse_table.py（V2=1）。語板は**語板 v2**（実験73、観点 9 = どんな・を・が・で・に・の前・の後・と・は、名詞 3 万語）。v1（観点 5）の表は [diff_reverse_table.md](diff_reverse_table.md)。"
for kind, pairs, nview in ((("名詞", NOUN_PAIRS, 9),) if V2 else (("名詞", NOUN_PAIRS, 5), ("動詞", VERB_PAIRS, 4))):
    words, B3, views, names = load_v2() if V2 else load(kind, nview); WI = {w: i for i, w in enumerate(words)}
    NVAL = B3.shape[-1] - nview                                                              # 値の出口の数（v1 は 47、v2 は 45）
    other = torch.zeros(nview, NVAL + nview, dtype=torch.bool); other[:, NVAL:] = True; other = other.reshape(-1)   # 観点ごとの「その他」の出口
    nodes = tree(B3); flat = B3.reshape(len(words), -1)
    out += [f"## {kind}（語板 {len(words)} 語、属性樹の節 {len(nodes)}）", "", "| A − B | 光の像（強い所） | 逆から入れて反応した語 | 属性樹の幹（道筋） | 幹の仲間 |", "|---|---|---|---|---|"]
    n_ok = 0
    for pr in pairs.split("|"):
        a0, b0 = pr.split(); a, b = (x if x in WI else norm(x) for x in (a0, b0))
        if a not in WI or b not in WI:
            out.append(f"| {a0} − {b0} | （語板がない: {', '.join(x for x in (a, b) if x not in WI)}） | | | |"); continue
        D = (flat[WI[a]] - flat[WI[b]]).clamp(min=0); D[other] = 0                            # 光の像の正の所（「その他」は中身が分からないので使わない）
        top_attr = [names.get(divmod(int(k), NVAL + nview), "その他") for k in torch.argsort(D, descending=True)[:3]]
        fq = torch.tensor([math.log(2 + FREQ[kind].get(w, 0)) for w in words])
        sc = (flat @ D) * fq; sc[WI[a]] = sc[WI[b]] = -1                                     # 逆から入れて戻る光（重なり）× よく出る度合い（P(語 | 像) ∝ P(像 | 語) P(語)）
        resp = [words[int(k)] for k in torch.argsort(sc, descending=True)[:5]]
        cand = [j for j, nd in enumerate(nodes) if nd["depth"] > 0]
        nsc = torch.tensor([float(torch.nn.functional.cosine_similarity(nodes[j]["diff"], D, dim=0)) for j in cand])
        best = nodes[cand[int(nsc.argmax())]]
        trunk = " → ".join(names.get(p, "?") for p in best["path"])
        mem = best["members"]; ms = torch.nn.functional.cosine_similarity(flat[mem], D[None], dim=1)
        mates = [words[int(mem[k])] for k in torch.argsort(ms, descending=True)[:7] if words[int(mem[k])] not in (a, b)][:5]
        out.append(f"| {a0} − {b0} | {'・'.join(top_attr)} | {'・'.join(resp)} | {trunk} | {'・'.join(mates)}（{len(mem)} 語） |"); n_ok += 1
    out.append(""); print(kind, n_ok, "組", flush=True)
open("docs/diff_reverse_table_v2.md" if V2 else "docs/diff_reverse_table.md", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("\n".join(out[:30]))
