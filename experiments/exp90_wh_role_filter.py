"""実験90: 疑問 → 答えの役割のフィルター（小さな文法の学習器、持ち主の選んだ 2 番）

着想: 2026-11-02 持ち主「では、2 からおねがい」。実験89 で、答えを探す仕組みのうち手で決めていたのは「疑問の語 → 答えの役割」の表だけだった。
      これを、材料から自動で作った質問で学ぶ小さな光の器（フィルター）に置き換える。
材料: 非公開フォルダの tools/gen_wh_questions.py が作った 44,650 問（3 万文を係り受けで木にし、述語に係る節を 1 つ隠して質問にしたもの）。
      直接の質問（浜辺で → どこで）と、のは の質問（亀をいじめていたのは誰ですか: 助詞が消えるので、残りの節から役割を推す）。文ごとに 9:1 で学習・確かめ。
入力の光（特徴ごとに 1 本の周波数を点ける）: 疑問の語・その後の助詞・のは の印・述語が受け身か・残りの節の役割（それぞれ 1 本）・述語の語（よく出る 300）。
出力: 答えの役割（よく出る 24 + その他）の割合。器 = 縦 512 × 板 16（全部の質問で共通の 1 つ）、参照の光あり（実験85 の気づき）。
比べ: 表（疑問の語・助詞・のは の組 → 役割、なければ疑問の語だけ → 全体）、光を使わない 1 層の重み、実験87・89 の手の表。
最後に、学んだフィルターを実験89 の「上から」の答え探しに入れて、浦島太郎の質問 30 問で手の表と比べる。
結果: results/exp90.txt。実行: python experiments/exp90_wh_role_filter.py
"""
import collections, json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import init_plates, _h, VAL0

sys.stdout.reconfigure(encoding="utf-8"); torch.manual_seed(0); PRIV = "D:/AI_git/Sin-CNN3_private"
DATA = json.load(open(f"{PRIV}/data/views/wh_questions.json", encoding="utf-8"))
lab_c = collections.Counter(d["label"] for d in DATA if d["split"] == "train")
LABS = [l for l, _ in lab_c.most_common(24)] + ["その他"]; LI = {l: i for i, l in enumerate(LABS)}; NLB = len(LABS)
verbs = [v for v, _ in collections.Counter(d["verb"] for d in DATA if d["split"] == "train").most_common(300)]
FEATS = (["wh:" + w for w in ("誰", "何", "どこ", "いつ", "どんな")] + ["part:" + p for p, _ in collections.Counter(d["part"] for d in DATA).most_common(20)]
         + ["cleft", "passive"] + ["other:" + r for r in LABS] + ["verb:" + v for v in verbs])
FI = {f: i for i, f in enumerate(FEATS)}; NF = len(FEATS)


def feats(d):
    f = ["wh:" + d["wh"], "part:" + d["part"]] + (["cleft"] if d["cleft"] else []) + (["passive"] if d["passive"] else [])
    f += ["other:" + (r if r in LI else "その他") for r in d["others"]] + ["verb:" + d["verb"]]
    return [FI[x] for x in f if x in FI]


def tens(rows):
    X = torch.zeros(len(rows), NF); y = torch.tensor([LI.get(d["label"], NLB - 1) for d in rows])
    for i, d in enumerate(rows):
        for j in feats(d):
            X[i, j] = 1.0
    return X, y


TR = [d for d in DATA if d["split"] == "train"]; TE = [d for d in DATA if d["split"] == "test"]
Xtr, ytr = tens(TR); Xte, yte = tens(TE)
kind_te = torch.tensor([d["cleft"] for d in TE])
print("学習", len(TR), "確かめ", len(TE), "特徴", NF, "役割", NLB, flush=True)
acc = lambda P, y, m=None: float(((P.argmax(1) == y) if m is None else (P.argmax(1) == y)[m]).float().mean())
res = dict(exp="90", train=len(TR), test=len(TE), labels=LABS)
# ---- 比べ: 表 ----
tab = collections.defaultdict(collections.Counter); tab_wh = collections.defaultdict(collections.Counter); glob = collections.Counter()
for d in TR:
    k = (d["wh"], d["part"], d["cleft"]); l = LI.get(d["label"], NLB - 1)
    tab[k][l] += 1; tab_wh[(d["wh"], d["cleft"])][l] += 1; glob[l] += 1
Pt = torch.zeros(len(TE), NLB)
for i, d in enumerate(TE):
    c = tab.get((d["wh"], d["part"], d["cleft"])) or tab_wh.get((d["wh"], d["cleft"])) or glob
    for l, m in c.items():
        Pt[i, l] = m
res["table"] = dict(all=acc(Pt, yte), direct=acc(Pt, yte, kind_te == 0), cleft=acc(Pt, yte, kind_te == 1))
# ---- 比べ: 1 層の重み（光なし） ----
Wl = torch.zeros(NF, NLB, requires_grad=True); bl = torch.zeros(NLB, requires_grad=True); opt = torch.optim.Adam([Wl, bl], lr=0.05)
for st in range(1500):
    idx = torch.randint(0, len(Xtr), (512,)); loss = torch.nn.functional.cross_entropy(Xtr[idx] @ Wl + bl, ytr[idx])
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    Pl = torch.softmax(Xte @ Wl + bl, 1)
res["linear"] = dict(all=acc(Pl, yte), direct=acc(Pl, yte, kind_te == 0), cleft=acc(Pl, yte, kind_te == 1))
# ---- 光のフィルター ----
N, K = 512, 16; H = _h("cpu", N); yy = torch.arange(N, dtype=torch.float32)
WAVE = torch.stack([torch.exp(1j * 2 * math.pi * (k + 1) * yy / N) / math.sqrt(N) for k in range(NF + 1)]).to(torch.complex64)   # 特徴 k = 周波数 k + 1、最後 = 参照の光


def forward(P, X):
    u = X.to(torch.complex64) @ WAVE[:NF]
    u = u / u.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-9)
    u = (u + WAVE[NF][None]) / math.sqrt(2)                                                  # 参照の光
    u = torch.nn.functional.pad(u, (0, N))
    for k in range(K):
        u = torch.cat([u[..., :N] * torch.exp(1j * P[:, k]), u[..., N:]], -1)
        u = torch.fft.ifft(torch.fft.fft(u, dim=-1, norm="ortho") * H, dim=-1, norm="ortho")
    I = torch.fft.fft(u[..., :N], dim=-1, norm="ortho").abs() ** 2
    o = I[:, VAL0:VAL0 + NLB]
    return o / o.sum(-1, keepdim=True).clamp(min=1e-12)


P = init_plates(1, 0, "cpu", N, K)[0].requires_grad_(True)
opt = torch.optim.Adam([P], lr=0.02); STEPS = 3000; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
for st in range(STEPS):
    idx = torch.randint(0, len(Xtr), (256,)); o = forward(P, Xtr[idx])
    loss = -torch.log(o.gather(1, ytr[idx][:, None])[:, 0] + 1e-9).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if st % 500 == 0:
        print(st, "回", round(float(loss), 4), flush=True)
with torch.no_grad():
    Po = torch.cat([forward(P, Xte[i:i + 2000]) for i in range(0, len(Xte), 2000)])
res["light"] = dict(all=acc(Po, yte), direct=acc(Po, yte, kind_te == 0), cleft=acc(Po, yte, kind_te == 1))
torch.save(dict(P=P.detach(), feats=FEATS, labels=LABS), f"{PRIV}/models/wh_role_filter.pt")
print("表", res["table"], "1 層", res["linear"], "光", res["light"], flush=True)
# ---- 浦島太郎の質問に入れる（実験89 の「上から」、手の表の代わりにフィルターの役割の割合を使う） ----
spec_path = os.path.join(os.path.dirname(__file__), "exp89_buffer_loop.py")
src = open(spec_path, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec_path}; exec(compile(src, spec_path, "exec"), ns)
role89, CONTENT = ns["build"].__globals__["role"], ns["CONTENT"]   # 実験87 の役割の決め方（質問の節にも同じものを使う）


def q_feats(q):
    """浦島太郎の質問から、フィルターの入力（疑問の語・助詞・のは・受け身・残りの節の役割・述語）を取る。"""
    txt = q["sentence"]; B = q["bunsetsu"]
    wh = next((w for w in ("誰", "どこ", "いつ", "どんな") if w in txt), "何")
    whb = next((b for b in B if any(x in b["text"] for x in ("誰", "何", "どこ", "いつ", "どんな", "名前"))), None)
    part = next((w[0] for w in whb["words"] if w[2].startswith("助詞")), "") if whb else ""
    cleft = int("のは" in txt)
    root = next(b for b in B if b["head"] == -1)
    verb_b = root if not cleft else next((b for b in B if "のは" in b["text"]), root)
    verb = next((w[1] for w in verb_b["words"] if w[2].startswith(("動詞", "名詞", "形容詞"))), "")
    passive = any(w[1] in ("れる", "られる") for w in verb_b["words"])
    others = [role89(b) for b in B if b is not whb and b is not root and b is not verb_b]
    if "名前" in txt:
        wh, part = "何", "という"                                                             # 「名前は」= という で答える（ここは表のまま）
    return dict(wh=wh, part=part if not cleft else "", cleft=cleft, passive=passive, others=others, verb=verb)


ROLES89, RI89, VOC = ns["ROLES"], ns["RI"], ns["VOC"]
orig_question = ns["question"]


def question_filter(q):
    b, r_hand = orig_question(q)
    d = q_feats(q)
    x = torch.zeros(1, NF)
    for j in feats(dict(d, label="")):
        x[0, j] = 1.0
    with torch.no_grad():
        p = forward(P, x)[0]
    r = torch.zeros_like(r_hand)
    for l, pr in zip(LABS, p.tolist()):
        if l in RI89:
            r[len(VOC) + RI89[l]] = pr                                                        # 答えの役割の光 = フィルターの割合（手の表の代わり）
    if d["part"] == "という":
        r = r_hand
    return b, r


R_hand = ns["run_all"](ns["Q1"], ns["A1"]) + ns["run_all"](ns["Q2"], ns["A2"])
ns["question"] = question_filter
R_filt = ns["run_all"](ns["Q1"], ns["A1"]) + ns["run_all"](ns["Q2"], ns["A2"])
res["urashima"] = dict(hand_20=sum(r["上から_ok"] for r in R_hand[:20]), hand_10=sum(r["上から_ok"] for r in R_hand[20:]),
                       filter_20=sum(r["上から_ok"] for r in R_filt[:20]), filter_10=sum(r["上から_ok"] for r in R_filt[20:]),
                       rows=[dict(q=a["q"], gold=a["gold"], hand=a["上から"], filt=b_["上から"], hand_ok=a["上から_ok"], filt_ok=b_["上から_ok"]) for a, b_ in zip(R_hand, R_filt)])
print("浦島太郎", {k: v for k, v in res["urashima"].items() if k != "rows"})
with open("results/exp90.txt", "a", encoding="utf-8") as f:
    f.write(json.dumps(res, ensure_ascii=False) + "\n")
