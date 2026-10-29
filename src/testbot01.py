"""testbot01.py — テストボット01: 光の語板（ごばん）で連想して、決まった型の文で返す、いちばん小さな会話ボット

持ち主の依頼（2026-10-29）: 「現状の学習結果から会話ボットを組み立てられる？ 次の試験で使いたい、テストボット01 として」。

■ 何ができて、何ができないか（正直に）
  できる: 入力の文を語に分け（形態素解析器 Sudachi）、名詞・動詞を見つけ、語板（1 語 = 1 枚の光の器）で「その語がどう使われるか」を連想して答える。
          - 名詞だけ → その名詞のいちばん自信のある使われ方を話題にする（例: ラーメン → 「食べる」）
          - 名詞 + を/が/で/に + 動詞 → その組み合わせが自然か（よくある / 珍しい = 驚き、学び 27）を判断し、
            動詞の語板で「どこで」「誰が」をたどって返す（名詞 → 動詞 → 名詞、学び 56・実験50）
          - 「〜ない」が付いた否定 → 残りの光（1 − p、学び 53）で「では何をするか」を返す
          - 知らない名詞 → 「知らない言葉です」と聞き返す（学び 51: 知らない物に気づく）
          - 自信が低いとき → 「たぶん」を付ける（学び 62: 光の偏り = 確信度）
  できない: 文を自由に作ること（返事はすべて決まった型）、前の会話を覚えること、語順・助詞以外の文法、名詞と動詞以外の理解。
           学んだ材料に Web 小説（研究のみ）を含むので、このボットと語板は公開しない。

■ 使い方
    python src/testbot01.py                     # 対話（終わるときは空行）
    from testbot01 import TestBot01
    bot = TestBot01(); print(bot.reply("ラーメンが好きです"))
    print(bot.explain("ラーメンを食べた"))        # 中で何を考えたか（語板の明るさ）を見る

■ 中身の流れ（reply の中）
    1. 文を語に分ける → 2. 名詞・助詞・動詞の並びを探す → 3. 語板に光を通して、観点ごとの値の明るさ（割合）を読む
    → 4. 場合分けして型の文を選ぶ → 5. 自信が低ければ「たぶん」を付ける
"""
import math, os, re, sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from goban_batch import run                                                                # 語板（光の器）に観点の光を通す計算

MODEL = "D:/AI_git/Sin-CNN3_private/models/testbot01.pt"                                    # 実験69 で作った語板（非公開フォルダ）

# 語板の値は「正規形」（例: 為る・御飯）なので、返事ではふつうの書き方に直す
KANA = {"為る": "する", "有る": "ある", "居る": "いる", "成る": "なる", "遣る": "やる", "出来る": "できる", "無い": "ない", "良い": "いい", "言う": "言う",
        "御飯": "ご飯", "御店": "お店", "御自分": "ご自分", "御金": "お金", "御茶": "お茶", "御酒": "お酒", "御家": "お家", "掛ける": "かける", "付く": "つく",
        "旨い": "うまい", "奇麗": "きれい", "仕舞う": "しまう", "御菓子": "お菓子", "見詰める": "見つめる", "御風呂": "お風呂", "嵌まる": "はまる", "茶": "お茶", "嫌": "嫌い", "御腹": "お腹", "御父さん": "お父さん", "御母さん": "お母さん"}
kana = lambda w: KANA.get(w, w)                                                            # 正規形 → 返事で使う書き方


class TestBot01:
    def __init__(self, path=MODEL):
        m = torch.load(path, weights_only=False)                                            # 語板の位相の板と、語・値の並びを読む
        self.NVIEW, self.VVIEW = m["NVIEW"], m["VVIEW"]                                     # 名詞の観点（どんな・を・が・で・に）、動詞の観点（を・が・で・に）
        self.NOUNS, self.VERBS = m["NOUNS"], m["VERBS"]                                     # 語板を持っている名詞・動詞
        self.NI = {n: i for i, n in enumerate(self.NOUNS)}; self.VI = {v: i for i, v in enumerate(self.VERBS)}   # 語 → 語板の番号
        self.Pn, self.Pv = m["P_名詞"], m["P_動詞"]                                           # 位相の板（語の数 × 縦 64 × 板 8 枚）
        self.NPOS, self.VPOS = m["NPOS"], m["VPOS"]                                         # 観点ごとの「値 → 出口の場所」
        self.nslots = 47 + len(self.NVIEW); self.vslots = 47 + len(self.VVIEW)              # 出口の数（値 47 個 + 観点ごとの「その他」）
        self.ncount = m["noun_count"]                                                       # 名詞のよく出る度合い（逆引きの重みに使う）
        from sudachipy import Dictionary, SplitMode                                         # 形態素解析器（語に分ける）
        self.tok = Dictionary(dict="core").create(); self.mode = SplitMode.C
        self._cache = {}                                                                    # 一度光を通した語板の答えを覚えておく（同じ計算をしない）

    # ---------- 語板に光を通す ----------
    def _profile(self, kind, word):
        """語板 word に観点の光を通し、観点ごとに {値: 割合} を返す（kind = "名詞" / "動詞"）。"""
        key = (kind, word)
        if key in self._cache:
            return self._cache[key]
        P, idx, views, pos, slots = (self.Pn, self.NI, self.NVIEW, self.NPOS, self.nslots) if kind == "名詞" else (self.Pv, self.VI, self.VVIEW, self.VPOS, self.vslots)
        with torch.no_grad():
            W = run(P[idx[word]][None], len(views), slots)[0]                               # (観点, 出口) の明るさ = 光の器の答え
        out = {}
        for j, v in enumerate(views):
            total = float(W[j].sum()) + 1e-12                                               # その観点で出口に届いた光の合計
            d = {x: float(W[j, s]) / total for x, s in pos[v].items()}                      # 値ごとの明るさを割合に直す
            d["(その他)"] = float(W[j, 47 + j]) / total                                     # 上位 47 個以外にまとめた分
            out[v] = d
        self._cache[key] = out
        return out

    def _top(self, dist, k=3, skip=("(その他)", "為る", "有る", "居る", "成る", "よう", "無い", "そう", "みたい", "様")):
        """割合の高い値を k 個（意味の薄い値・その他は飛ばす）。"""
        return [(x, p) for x, p in sorted(dist.items(), key=lambda t: -t[1]) if x not in skip][:k]

    def _confidence(self, dist):
        """自信 = いちばん明るい値の割合（学び 62: 光が 1 か所に集まるほど自信が高い）。"""
        return max(p for x, p in dist.items() if x != "(その他)")

    # ---------- 文を読む ----------
    def _parse(self, text):
        """文を語に分け、(表層, 正規形, 品詞1, 品詞2) の並びにする。"""
        return [(m.surface(), m.normalized_form(), *m.part_of_speech()[:2]) for m in self.tok.tokenize(text, self.mode)]

    def _find(self, ms):
        """名詞 + 助詞（を・が・で・に）+ 動詞 の組、ひとりの名詞、否定・知らない名詞を探す。"""
        triples, nouns, unknown = [], [], []
        for i, m in enumerate(ms):
            if m[2] != "名詞" or m[3] not in ("普通名詞", "固有名詞"):
                continue
            if m[1] not in self.NI:                                                         # 語板を持っていない名詞 = 知らない言葉
                unknown.append(m[0]); continue
            nouns.append(m[1])
            if i + 2 < len(ms) and ms[i + 1][2] == "助詞" and ms[i + 1][0] in ("を", "が", "で", "に") and ms[i + 2][2] in ("動詞", "形容詞"):
                rest = "".join(x[0] for x in ms[i + 3:i + 6])                               # 動詞のすぐ後ろ（「〜ない」「〜ません」を見る）
                neg = bool(re.match(r"^(ない|なかっ|ません|ず|ぬ)", rest)) or ms[i + 2][0].endswith("ない")
                triples.append((m[1], ms[i + 1][0], ms[i + 2][1], neg))
        for i, m in enumerate(ms):                                                          # 「〜は好き」「〜が美味しい」のような形容詞・形状詞も「どんな」の手がかりにする
            pass
        return triples, nouns, unknown

    # ---------- 返事を作る ----------
    def reply(self, text):
        ms = self._parse(text); triples, nouns, unknown = self._find(ms)
        if triples:
            return self._reply_triple(*triples[0])                                          # 組（名詞 + 助詞 + 動詞）があれば、それを話題にする
        if nouns:
            return self._reply_noun(nouns[0])                                               # 名詞だけなら、その名詞の使われ方を話題にする
        if unknown:
            return f"「{unknown[0]}」は、まだ知らない言葉です。どんなものか教えてもらえますか？"   # 知らない物に気づく（学び 51）
        return "なるほど。もう少し詳しく聞かせてください。"                                     # 何も見つからない

    def _hedge(self, s, conf, th=0.25):
        """自信が低いときは「たぶん」を付ける（学び 62）。"""
        return ("たぶんですが、" + s) if conf < th else s

    def _reply_noun(self, n):
        prof = self._profile("名詞", n)                                                     # 名詞の語板に、5 つの観点の光を通す
        conf_of = lambda v: self._confidence({x: p for x, p in prof[v].items() if x not in ("為る", "有る", "成る", "居る", "よう", "無い", "そう", "みたい", "様")})
        best_v = next((v for v in ("を", "どんな", "が", "に", "で") if conf_of(v) >= 0.2), None)   # 話題にしやすい観点から順に、自信が 0.2 以上のものを選ぶ
        best_v = best_v or max(("を", "どんな", "が", "に", "で"), key=conf_of)                  # どれも低ければ、いちばん自信のある観点
        tops = self._top(prof[best_v]); conf = tops[0][1] if tops else 0.0                   # いちばん自信のある観点の、上位の値
        if not tops:
            return f"{kana(n)}ですか。もう少し聞かせてください。"
        x = kana(tops[0][0]); nn = kana(n)
        if best_v == "を":
            s = f"{nn}といえば、{x}ことが多いですよね。最近{x}機会はありましたか？"                # 動詞は辞書の形のまま使う（活用は作らない）
        elif best_v == "で":
            s = f"{nn}で{x}ことが多いんですか？"
        elif best_v == "に":
            s = f"{nn}に{x}ことってありますか？"
        elif best_v == "が":
            s = f"{nn}が{x}こと、ありますよね。"
        else:
            adj = x if x.endswith("い") else x + "な"                                      # 形容詞はそのまま、形状詞（好き・きれい）は「な」を付ける
            s = f"{adj}{nn}、いいですね。"
        return self._hedge(s, conf)

    def _reply_triple(self, n, part, verb, neg):
        prof = self._profile("名詞", n)[part]                                               # 「名詞 + 助詞」の観点で、動詞の割合を読む
        known = verb in prof                                                                # 動詞が語板の値（上位 47 個）に入っているか
        p = prof[verb] if known else None                                                   # 入っていなければ、割合は分からない（「その他」の中）
        nn, vv = kana(n), kana(verb)
        others = [x for x, _ in self._top(prof) if x != verb]
        if neg:                                                                             # 否定: 残りの光（1 − p）から、代わりによくある使い方を出す（学び 53）
            alt = kana(others[0]) if others else "ほかのこと"
            return f"そうなんですね。{nn}は、{alt}ことのほうが多いみたいですよ。"
        if known and p < 0.01:                                                              # 珍しい組み合わせ = 驚き（学び 27: 合わない文は光が暗い）
            alt = kana(others[0]) if others else "ほかのこと"
            return f"{nn}{part}{vv}って、ちょっと珍しいですね！ふつうは{alt}ことが多いみたいです。"
        place = None                                                                        # よくある組み合わせ: 動詞の語板で「どこで」をたどる（名詞 → 動詞 → 名詞）
        if verb in self.VI:
            vp = self._profile("動詞", verb)["で"]
            cand = [(x, q * math.log(2 + self.ncount.get(x, 1))) for x, q in vp.items() if x != "(その他)" and x != n]   # 逆引きはよく出る度合いを掛ける（学び 59・61）
            if cand:
                place = kana(max(cand, key=lambda t: t[1])[0])
        if not known:                                                                       # 値にない動詞: よくあるか珍しいか分からないので、決めつけずに聞き返す
            return f"そうなんですね、{nn}{part}{vv}んですね。" + (f"{place}で{vv}ことが多いんですか？" if place else "どんな感じですか？")
        s = f"いいですね！{nn}{part}{vv}んですね。" + (f"{place}で{vv}ことが多いんですか？" if place else "")
        return self._hedge(s, p, th=0.05)

    def explain(self, text):
        """中で何を考えたか: 見つけた語・組と、語板の上位の値と割合。"""
        ms = self._parse(text); triples, nouns, unknown = self._find(ms)
        info = dict(語=[m[0] for m in ms], 組=triples, 名詞=nouns, 知らない名詞=unknown)
        for n in nouns[:2]:
            info[f"語板「{n}」"] = {v: [(kana(x), round(p, 3)) for x, p in self._top(d, 3, skip=("(その他)",))] for v, d in self._profile("名詞", n).items()}
        return info


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    bot = TestBot01()
    print("テストボット01（終わるときは空行）")
    while True:
        try:
            s = input("あなた> ").strip()
        except EOFError:
            break
        if not s:
            break
        print("ボット>", bot.reply(s))
