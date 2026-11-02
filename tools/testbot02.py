"""テストボット02: 返事を選ぶ会話ボット（実験115 の選び方 C: 語の重なり + 次の発言の共鳴）+ 結びは基本丁寧（実験117 の付け替え）
+ 感情 → 内容 → 質問 の型（バックトラッキング = 相手の述語を「〜んですね」で返す、持ち主が調べた会話継続のテクニック）。
持ち主との方針（2026-11-02）: いきなり文を一から作らず、会話の材料の中から返事を選ぶ。生成は少しずつ置き換える。
返事の候補: RealPersonaChat の雑談の発言（nu-dialogue、CC BY-SA 4.0）。案内の対話（JMultiWOZ）は雑談に合わないので候補から外す。
  手順: 1. 相手の発言の中身の語 → 2. 共鳴の強い「次に出やすい語」を引く（逆引きの索引）→ 3. その語を含む発言を候補にする
       → 4. 実験115 の点（語の重なり + 共鳴）で並べる → 5. 口調（相手が丁寧なら丁寧、くだけていればくだけた）が合うものを優先 → 上位から 1 つ。
  「自信」: 1 位の点が低いとき（共鳴の手がかりが無いとき）は「よく分からない」と正直に返す（実験62 の考えの最初の形）。
実行: python tools/testbot02.py "発言 1" "発言 2" …（引数が無ければ、決まった例で動かす）
"""
import json, math, random, re, sys, zlib

import torch
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.argv, ARGS = [sys.argv[0]], sys.argv[1:]
src = open("experiments/exp115_response_selection.py", encoding="utf-8").read().split("# ---- 試し: 10 択 ----")[0]
exec(src)                                                                                      # 材料・解析・共鳴の数え上げ（実験115）
# ---- 候補の発言と、逆引きの索引（語 → その語を含む発言）----
POOL = [t for s, turns in convs["train"] if s == "real-persona-chat" for t in turns if 4 <= len(t) <= 60]
POOL = list(dict.fromkeys(POOL))
INV = defaultdict(list)
for k, t in enumerate(POOL):
    for w in analyze(t)[0]:
        INV[w].append(k)
NEXT = defaultdict(list)                                                                       # 速くするため、先に語ごとの上位を作る
for (x, y), c in cpair.items():
    if c >= 3:
        NEXT[x].append((c, y))
for x in NEXT:
    NEXT[x] = [y for c, y in sorted(NEXT[x], reverse=True)[:60]]
print("候補の発言", len(POOL), flush=True)


# ---- 状態遷移グレイン（実験120）: 相手の発言の状態 → 次に返す状態 の見込み（持ち主「次にどの状態で返すかを決める」）----
TG = torch.load("D:/AI_git/Sin-CNN3_private/models/transition_grain_120.pt", weights_only=False)
AIZ = ("そう", "へー", "へえ", "なるほど", "確かに", "たしかに", "いいですね", "わかります", "分かります", "ですよね", "ほんと", "本当")
EMO_F = ("嬉し", "楽し", "悲し", "辛", "つら", "大変", "疲れ", "驚", "困", "怖", "寂し", "面白", "美味し", "好き", "嫌", "心配", "残念", "最高", "幸せ", "羨まし", "すごい", "凄い")
SELF = ("私", "僕", "自分", "うち", "俺", "わたし"); YOU = ("さん", "あなた", "そちら")
GREET_F = ("こんにちは", "こんばんは", "おはよう", "よろしく", "ありがとう", "失礼", "さようなら", "またね", "お疲れ")


def tg_state(prev, t):
    """発言の形の手がかり（実験120 と同じ 26 個）→ いちばん近い状態の番号。"""
    tt = t.rstrip(); pw = analyze(prev)[0] if prev else set(); cw_ = analyze(t)[0]
    q = tt.endswith(("？", "?", "か", "か。", "の？"))
    v = [float(tt.endswith(e)) for e in TG["ENDS"]]
    v += [float(tt.startswith(AIZ)), float(any(e in tt for e in EMO_F)), float(any(tt.startswith(x) or (x + "は") in tt for x in SELF)),
          float(any(y in tt for y in YOU)), float(q), float(bool(pw & cw_)), float(bool(pw) and not (pw & cw_)),
          min(len(tt), 80) / 80, float(any(g in tt[:12] for g in GREET_F)), float("？" in tt[:-1] or "?" in tt[:-1])]
    z = (torch.tensor([v]) - TG["mu"]) / TG["sd"]
    return int(torch.cdist(z, TG["C"]).argmin(1))


# ---- キャラクタのプロフィール（持ち主が書いたもの、data/characters/）: 自己紹介・自分についての質問・返事選びの軽いバイアス ----
CHAR_PATH = next((a.split("=", 1)[1] for a in ARGS if a.startswith("--char=")), "data/characters/secretary_kotoyui_tonari.json")
ARGS = [a for a in ARGS if not a.startswith("--char=")]
PROF = json.load(open(CHAR_PATH, encoding="utf-8"))
PBIAS = set(analyze(" ".join(PROF.get("バイアスの語", []) + [PROF.get("興味のある内容", ""), PROF.get("好きな会話", ""), PROF.get("今の目標", "")]))[0])
PW = PROF.get("バイアスの強さ", 0.2)
OTHER_SELF = [x for x in ("僕", "俺", "あたし", "わし", "おいら", "わたし", "私") if x != PROF.get("一人称", "私") and not (PROF.get("一人称") == "私" and x == "わたし")]


def self_answer(user, history):
    """自分についての質問・自己紹介への答え（プロフィールの項目を、決まった言い回しに入れて返す）。当てはまらなければ None。"""
    u = user
    if len(history) <= 1 and any(g in u for g in ("こんにちは", "はじめまして", "初めまして", "よろしく", "こんばんは", "おはよう")):
        intros = PROF["自己紹介"]
        return intros[2] if "名前" not in u and len(intros) > 2 else intros[0]               # 始めのあいさつ → 自己紹介（相手の名前を聞く形を優先）
    if "自己紹介" in u:
        return PROF["自己紹介"][0]
    to_me = isq(u) or any(x in u for x in ("あなた", "君", "きみ", "となり", PROF["苗字"]))   # 自分についての質問か（ただの話の中の「仕事」には答えない）
    if not to_me:
        return None
    if any(x in u for x in ("名前", "誰", "だれ")) and any(x in u for x in ("あなた", "君", "きみ", "お名前", "名前は", "誰", "だれ")):
        return f"{PROF['一人称']}は{PROF['苗字']}{PROF['名前']}と申します。{PROF['苗字']}と呼んでください。"
    if any(x in u for x in ("好きな", "好き？", "何が好き")):
        return f"{PROF['好きな会話']}が好きです。色々教えてもらえると嬉しいです！"
    if any(x in u for x in ("嫌いな", "苦手")):
        return f"{PROF['嫌いな会話']}は、少し苦手です。"
    if any(x in u for x in ("興味", "趣味")):
        return f"{PROF['興味のある内容']}に興味があります。お話を聞いて、一緒に考えるのが好きなんです。"
    if any(x in u for x in ("何してる", "仕事", "目標", "何ができる", "何をしてる")):
        return f"今は、お話しする方の{PROF['役割']}のようなお手伝いができるようになりたいと思っています。"
    if any(x in u for x in ("どうやって", "どこから来", "生まれ", "仕組み")):
        return f"{PROF['過去・生い立ち']}んです。{PROF['長所']}なところが自慢です。"
    if any(x in u for x in ("弱点", "短所", "苦手なこと", "できないこと")):
        return f"{PROF['短所']}ので、間違えたら教えてください。"
    return None


def candidates(history):
    uw, up = analyze(history[-1])
    s_user = tg_state(history[-2] if len(history) >= 2 else "", history[-1])                  # 相手の発言の状態
    keys = set(uw) | {y for x in uw for y in NEXT.get(x, [])[:40]}
    cand = Counter()
    for y in keys:
        for k in INV.get(y, [])[:3000]:
            cand[k] += 1
    ids = [k for k, _ in cand.most_common(400)]
    if not ids:
        return []
    scored = []
    for k in ids:
        t = POOL[k]
        if t in history:
            continue
        cw_, cp = analyze(t)
        f_over = len(uw & cw_) / math.sqrt(len(cw_) + 1)
        f_res = sum(pmi(x, y) for x in uw for y in cw_) / math.sqrt(len(uw) * len(cw_)) if uw and cw_ else 0.0
        f_tone = 1.0 if cp == up else 0.0
        f_tr = float(TG["logP"][s_user, tg_state(history[-1], t)])                             # この返事の状態は、相手の状態のあとに来やすいか
        if any(x in t for x in OTHER_SELF):
            continue                                                                           # キャラの一人称（私）と違う一人称の発言は選ばない
        f_prof = float(bool(cw_ & PBIAS))                                                       # キャラの興味・目標の語を含むか（軽いバイアス）
        scored.append((f_over + f_res + 0.3 * f_tone + 0.5 * f_tr + PW * f_prof, t, (f_over, f_res, f_tone, f_tr, f_prof)))   # 点・発言・手がかり（重なり・共鳴・口調・遷移・キャラ）
    scored.sort(key=lambda x: -x[0])
    return scored


def ensemble(scored, n=10, seed=0):
    """持ち主の案: 選ぶ重みを少しずつ揺らして 10 通り選び、票の多い返事を選ぶ。票の割合 = 自信。"""
    rng = random.Random(seed); votes = Counter()
    for _ in range(n):
        w = [math.exp(rng.gauss(0, 0.4)), math.exp(rng.gauss(0, 0.4)), 0.3 * math.exp(rng.gauss(0, 0.4)), 0.5 * math.exp(rng.gauss(0, 0.4)), PW * math.exp(rng.gauss(0, 0.4))]
        best = max(scored, key=lambda x: sum(wi * fi for wi, fi in zip(w, x[2])))
        votes[best[1]] += 1
    t, v = votes.most_common(1)[0]
    return t, v / n


# ---- 結びの付け替え（実験117）と、バックトラッキング（持ち主が調べた会話継続のテクニック 2026-11-02）----
#   ※ 尾の表は実験117 の材料（実験だけに使う材料を含む）から作っている。苗として配るときは、ライセンスの分かる材料で作り直す。
_src117 = open("experiments/exp117_ending_swap.py", encoding="utf-8").read().split('res = dict(exp="117")')[0]
_ns117 = {}
exec(_src117, _ns117)
swap, split117, peel, polite_tail = _ns117["swap"], _ns117["split"], _ns117["peel"], _ns117["polite"]
FUNC117 = _ns117["FUNC"]
EMO = {"感動", "嬉しい", "楽しい", "悲しい", "辛い", "大変", "疲れる", "驚く", "困る", "怖い", "寂しい", "面白い", "美味しい", "好き", "嫌い", "心配", "残念", "最高", "幸せ", "ショック", "緊張"}


isq = lambda t: t.rstrip().endswith(("？", "?", "か", "か。", "の？"))


def morphs(text):
    return [(m.surface(), m.normalized_form(), "-".join(x for x in m.part_of_speech() if x != "*")) for m in tok.tokenize(text)]


def last_predicate(ms):
    """文の最後の述語の文節（感情の語があればそれを優先）。"""
    idx = [i for i, w in enumerate(ms) if not w[2].startswith(FUNC117) and ("非自立可能" in w[2] or "非自立" not in w[2]) and w[2].startswith(("動詞", "形容詞", "形状詞", "名詞"))
           and not (i > 0 and ms[i - 1][2].startswith("助詞-接続助詞"))]
    emo = [i for i, w in enumerate(ms) if w[1] in EMO and not w[2].startswith(FUNC117)]       # 感情の語は「〜て」のあとでも拾う（忙しくて大変）
    return (emo or idx or [None])[-1]


# ---- 格逆算グレイン（実験119）: 動詞の格の好み P(名詞 | 動詞, 格)。省略された格を、会話に出た名詞から補う ----
import os, pickle
_FP = "D:/AI_git/Sin-CNN3_private/models/case_frames_119.pkl"
if os.path.exists(_FP):
    FR, FRN = pickle.load(open(_FP, "rb"))
else:
    FR = defaultdict(Counter); FRN = Counter()
    for _s, turns in convs["train"]:
        for t in turns:
            ms_ = list(tok.tokenize(t))
            for i, m in enumerate(ms_):
                pos = m.part_of_speech()
                if pos[0] == "名詞" and pos[1] in ("普通名詞", "固有名詞") and len(m.surface()) >= 2:
                    FRN[m.normalized_form()] += 1
                    if i + 1 < len(ms_) and ms_[i + 1].surface() in ("が", "を", "に") and ms_[i + 1].part_of_speech()[1] == "格助詞":
                        v = next((x for x in ms_[i + 2:i + 6] if x.part_of_speech()[0] == "動詞" and x.part_of_speech()[1] != "非自立可能"), None)
                        if v is not None:
                            FR[(v.normalized_form(), ms_[i + 1].surface())][m.normalized_form()] += 1
    pickle.dump((FR, FRN), open(_FP, "wb"))
_FRT = sum(FRN.values())


def recover_case(verb, history):
    """動詞に「を・に」が無いとき、会話に出た名詞の中から、格の好みでいちばん合う「名詞 + 格」を補う（強く合う時だけ）。"""
    nouns = []
    for u in history[-4:]:
        for m in tok.tokenize(u):
            if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("普通名詞", "固有名詞") and len(m.surface()) >= 2:
                nouns.append((m.normalized_form(), m.surface()))
    best = None
    for c in ("を", "に"):
        cnt = FR.get((verb, c))
        if not cnt:
            continue
        tot = sum(cnt.values())
        for n, surf in nouns:
            lift = (cnt[n] / tot) / ((FRN[n] + 1) / _FRT) if cnt[n] >= 2 else 0               # この動詞の格に、その名詞が全体よりどれだけ来やすいか
            if lift > 5 and (best is None or lift > best[0]):
                best = (lift, surf + c)
    return best[1] if best else ""


def backtrack(user, history=()):
    """「昨日、映画を見てすごく感動したよ」→「感動したんですね」。相手の述語（感情の語を優先）を、くだけた形 + んですね で返す。質問には返さない。
    「何を」が省かれていれば、前の会話から補う（「昨日食べたよ」→「ラーメンを食べたんですね」）。"""
    if isq(user) or any(g in user for g in GREET_F):
        return ""                                                                              # 問いとあいさつには「〜んですね」で受けない（遷移: あいさつ → あいさつ・問い）
    ms = morphs(user); k = last_predicate(ms)
    if k is None:
        return ""
    if ms[k][1] in ("為る", "する") and k > 0 and ms[k - 1][2].startswith("名詞"):
        k -= 1                                                                                 # 整理 + する は、整理 から（予定を整理したいんですね）
    j = k + 1
    while j < len(ms) and (ms[j][2].startswith(FUNC117) or "非自立" in ms[j][2]) and not ms[j][2].startswith("補助記号") and not (
            ms[j][2].startswith("助詞-接続助詞") and not (j + 1 < len(ms) and "非自立" in ms[j + 1][2])):   # 行って + みたい のように、て の後ろが補助の動詞なら続ける
        j += 1
    core, tail = split117(ms[k:j]); tail, _ = peel(tail)
    tail = [w for w in tail if not w[2].startswith("助詞-終助詞")]
    while tail and tail[-1][1] in ("だ", "です", "の", "ん") and not (tail[-1][1] == "だ" and tail[-1][0] in ("だっ",)):
        tail = tail[:-1]                                                                       # 〜んだ・〜のだ・〜で（説明・つなぎ）は外して「んですね」に付け替える
    lem = ms[k][1]; pos = ms[k][2]
    arg = argument(ms, k)                                                                      # 持ち主の案: 「何に・何が」どうした
    if pos.startswith("動詞") and (not arg or arg.endswith(("で", "と"))):
        rec = recover_case(lem, list(history))                                                 # 省かれた「を・に」を前の会話から補う（実験119）
        if rec:
            arg = rec
    if pos.startswith(("名詞", "形状詞")) and not [w for w in tail if w[1] not in ("だ", "で")]:
        return arg + "".join(w[0] for w in core) + "なんですね。"                             # 散歩が大変で → 散歩が大変なんですね
    if not tail:
        return arg + "".join(w[0] for w in core) + "んですね。"
    if polite_tail(tail):
        past = any(w[1] == "た" for w in tail); neg = any(w[1] in ("ない", "ん", "ぬ") for w in tail)
        out, _ = swap(core, tail, False)
        if out and past and out.endswith(("た", "だ")) and "ください" not in out:
            return arg + out + "んですね。"
        if not past and not neg:
            return arg + lem + "んですね。"                                                    # 北海道に行きます → 北海道に行くんですね（辞書の形）
        return ""
    return arg + "".join(w[0] for w in core + tail) + "んですね。"


def argument(ms, k):
    """述語のすぐ前の「名詞 + が・を・に・へ・で・と」（何に・何が）。は（主題）や 私・僕 は付けない。"""
    if k >= 2 and ms[k - 1][2].startswith("助詞-格助詞") and ms[k - 1][0] in ("が", "を", "に", "へ", "で", "と"):
        j = k - 2
        while j >= 0 and ms[j][2].startswith(("名詞", "接頭辞", "接尾辞")) and "副詞可能" not in ms[j][2]:   # 最近・今日 などは付けない
            j -= 1
        noun = "".join(w[0] for w in ms[j + 1:k - 1])
        if noun and ms[k - 2][1] not in ("私", "僕", "俺", "自分", "何", "事", "物"):
            return noun + ms[k - 1][0]
    return ""


def to_polite(text):
    """返事の最後の文の結びを丁寧に（基本は丁寧、持ち主 2026-11-02）。"""
    ms = morphs(text); k = last_predicate(ms)
    if k is None:
        return text
    core, tail = split117(ms[k:]); tail, fin = peel(tail)
    if not tail or polite_tail(tail) or text.rstrip().endswith(("？", "?")):
        return text
    out, _ = swap(core, tail, True)
    return "".join(w[0] for w in ms[:k]) + out + "".join(w[0] for w in fin) if out else text


# ---- 話題を続ける質問（持ち主の案「北海道 → 札幌ですか？」）: 同じ発言の中で一緒に出やすい名詞（属性先の語板の代わり、数え上げ）----
CO = defaultdict(Counter); NOUNF = Counter()
for t in POOL[:150000]:
    ns = {m.normalized_form() for m in tok.tokenize(t) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("固有名詞", "普通名詞") and len(m.surface()) >= 2}
    for a in ns:
        NOUNF[a] += 1
        for b in ns:
            if a != b:
                CO[a][b] += 1
print("名詞の一緒に出る表", len(CO), flush=True)


def related_question(user, history):
    """相手の名詞（固有名詞を優先）から、一緒に出やすい名詞を選んで「〜ですか？」と聞く。"""
    nouns = [m for m in tok.tokenize(user) if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("固有名詞", "普通名詞") and len(m.surface()) >= 2]
    nouns.sort(key=lambda m: (m.part_of_speech()[1] != "固有名詞", NOUNF[m.normalized_form()]))
    said = "".join(history)
    for m in nouns:
        if m.part_of_speech()[1] != "固有名詞":
            continue                                                                           # 普通名詞からの「〜ですか？」は外れやすい（挑戦ですか？）ので、固有名詞（北海道 → 札幌）だけ
        x = m.normalized_form(); c = CO.get(x)
        if not c or NOUNF[x] < 5:
            continue
        best = max(((v / math.sqrt(NOUNF[y] + 10), y) for y, v in c.items() if v >= 10 and y not in said and NOUNF[y] >= 5 and len(y) >= 2 and y not in STOP and not all("ぁ" <= ch <= "ゖ" for ch in y)), default=None)   # ひらがなだけの語（こと・もの）は外す
        if best:
            return f"{best[1]}ですか？"
    return None


SESSION_NEW = {}                                                                               # 聞いた話から作る語（バッファ）: 語 → 一緒に聞いた語


def unknown_word(user):
    """知らない語（教えの会話に出ない名詞）があれば聞き返す。聞いた話はバッファに残す（持ち主の案: 語板をリアルタイムに増やす）。"""
    for m in tok.tokenize(user):
        if m.part_of_speech()[0] == "名詞" and m.part_of_speech()[1] in ("固有名詞", "普通名詞") and len(m.surface()) >= 2:
            x = m.normalized_form()
            if cw.get(x, 0) + cnext.get(x, 0) == 0 and x not in SESSION_NEW:
                SESSION_NEW[x] = set(analyze(user)[0]) - {x}
                return m.surface()
    return None


USER_NAME = None
PROB_EMO = ("大変", "困", "つら", "辛", "疲れ", "忙し", "悩", "心配", "不安", "面倒", "無理", "ヤバ", "やば", "焦")
PROB_NEG = ("できな", "出来な", "終わらな", "間に合わな", "わからな", "分からな", "進まな", "うまくいかな", "上手くいかな", "決まらな", "足りな")   # 〜ない・〜なくて の両方


def problem_score(user):
    """相手が課題を抱えているか（持ち主の問い 2026-11-02）。困りごとの感情の語・したいのにの形・できないの形・言いさし を足す。"""
    sc = 0.0
    sc += 1.0 if any(e in user for e in PROB_EMO) else 0.0
    sc += 1.0 if re.search(r"たい(んだ|んです|ん)?(けど|が|のに)", user) else 0.0
    sc += 1.0 if any(n in user for n in PROB_NEG) else 0.0
    sc += 0.5 if re.search(r"(けど|けれど|が|のに)[。…]*$", user.strip()) else 0.0
    return sc


def secretary_offer(user, bt):
    """秘書の型の返事: 課題があるときは、中身を聞いて手伝いを申し出る。"""
    m_ = re.match(r"(.+?)[をにが]", bt) or re.search(r"([一-龥ァ-ヶー]{2,8})(?:が|を)", user)   # 何が大変か: バックトラッキングの「何に」か、相手の発言の「〜が・〜を」
    noun = m_.group(1) if m_ and 1 < len(m_.group(1)) <= 8 else None
    if noun:
        noun = re.sub(r"^(最近|今日|昨日|明日|今週|来週|先週|今)", "", noun) or None              # 時の語は外す（最近仕事 → 仕事）
    if re.search(r"たい(んだ|んです|ん)?(けど|が|のに)", user) and noun:
        return f"どんな{noun}がありますか？ よければ一緒に整理しましょう。"
    if noun:
        return f"{noun}のどんなところが一番大変ですか？ 何かお手伝いできることがあれば言ってくださいね。"
    return "どんなところが一番大変ですか？ 何かお手伝いできることがあれば言ってくださいね。"


def reply(history):
    """感情（「何に」どうした、のバックトラッキング）→ 内容（揺らした 10 通りの票で選ぶ）→ 質問（話題を続ける）。知らない語は聞き返す。"""
    global USER_NAME
    if len(history) >= 2 and "お名前" in history[-2] and not USER_NAME:                         # 名前を聞いたあとの「山田です」を覚える
        m_ = re.match(r"^(?:私は|僕は)?([^、。！!？?\s]{1,8}?)(?:です|と申します|といいます|だよ|と言います)", history[-1])
        if m_:
            USER_NAME = m_.group(1)
            rest = history[-1][m_.end():].lstrip("。、！! ")
            sa = self_answer(rest, history) if rest else None
            return f"{USER_NAME}さん、よろしくお願いします！" + (sa or ""), 1.0
    sa = self_answer(history[-1], history)
    if sa:                                                                                     # 自己紹介・自分についての質問はプロフィールから
        return sa, 1.0
    bt = backtrack(history[-1], history[:-1])
    interest = PROF.get("課題解決への興味", 0.0)                                                # プロフィールの特徴（持ち主の案）: 大きいほど課題に気づきやすい
    if interest > 0 and not isq(history[-1]) and problem_score(history[-1]) >= 1.0 / interest:
        return bt + secretary_offer(history[-1], bt), 1.0                                      # 課題がありそう → 中身を聞いて手伝いを申し出る
    new = unknown_word(history[-1])
    if new:                                                                                    # 知らない語は共感を弱める（持ち主 2026-11-02: 分からないまま「〜んですね」と受けると不自然）
        return f"「{new}」って何ですか？ 初めて聞きました。", 0.0
    scored = candidates(history)
    if not scored:
        return bt + "もう少し教えてもらえますか？", 0.0
    best, conf = ensemble(scored)
    parts = [bt]
    if conf < 0.5:                                                                             # 票が割れた = 自信がない → 内容は出さず、話題の名詞について「どんな〜」と聞く
        m_ = re.match(r"(.+?)[をにが]", bt)
        if m_ and 1 < len(m_.group(1)) <= 8:
            past = "た" in bt[m_.end():]
            return bt + f"どんな{m_.group(1)}{'でした' if past else 'です'}か？", conf
        rq = related_question(history[-1], history)
        if rq:
            return bt + rq, conf
        parts.append("（自信がないけど）")
    parts.append(to_polite(best))
    if not isq(best) and len(best) < 30:                                                        # 長い返事には質問を足さない
        rq = related_question(history[-1], history + [best])
        if rq:
            parts.append(rq)
    out = ""
    for p_ in parts:
        if p_:
            out += ("。" if out and not out.endswith(("。", "！", "？", "?", "!", "）", "w", "笑")) else "") + p_   # つなぎ目に句点
    return out.replace("？。", "？").replace("！。", "！"), conf


if ARGS[:1] == ["--chat"]:                                                                     # 手元で話す: python tools/testbot02.py --chat（空行で終わる）
    hist = []
    while True:
        try:
            u = input("人: ").strip()
        except EOFError:
            break
        if not u:
            break
        hist.append(u); r, sc = reply(hist); hist.append(r)
        print("ボット:", r, f"（自信 {sc:.1f}）", flush=True)
    sys.exit(0)
if ARGS[:1] == ["--dialog"]:                                                                   # 続いた会話の試し: 発言を順に渡す（前の会話が格の逆算・遷移に効く）
    hist = []
    for u in ARGS[1:]:
        hist.append(u); r, sc = reply(hist); hist.append(r)
        print("人:", u); print("ボット:", r, f"（自信 {sc:.1f}）", flush=True)
    sys.exit(0)
tests = ARGS or ["昨日、映画を見てすごく感動したよ", "こんにちは！最近ラーメンにはまってるんだ", "休みの日は何をしていますか？", "犬を飼っているんですが、散歩が大変で", "明日から旅行で北海道に行きます", "最近寝不足でつらいよ", "週末にポポロンタをやったよ"]
out = []
for u in tests:
    r, sc = reply([u]); out.append(dict(user=u, bot=r, confidence=round(sc, 2)))
    print("人:", u); print("ボット:", r, f"（自信 {sc:.1f}）"); print()
json.dump(out, open("results/testbot02_examples.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
