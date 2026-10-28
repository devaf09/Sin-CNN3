"""図: 実験35〜49 の結果（results/ の数字から描く）。1 つの実験 = 1 枚の図。
実行: python tools/fig_exp35_49.py → docs/images/exp35_order_colors.png 〜 exp49_attention.png"""
import collections, json, math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
load = lambda n: json.loads(open(f"results/exp{n}.txt", encoding="utf-8").readline())
lines = lambda n: [json.loads(l) for l in open(f"results/exp{n}.txt", encoding="utf-8")]
mm = lambda xs: (np.mean(xs), np.min(xs), np.max(xs))


def bars(ax, names, vals, colors, fmt="{:.1f}", lo=None, hi=None):
    ax.bar(range(len(names)), vals, color=colors, yerr=None if lo is None else [np.array(vals) - lo, np.array(hi) - vals], capsize=3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"docs/images/{name}.png"); plt.close(fig)


# 実験35: 直前 2 文字を別々の色 + カー効果
d = load(35)["rows"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=130)
names = [r["kind"].replace("（数える）", "").replace("光の器", "光").replace("曲がる処理なし", "線形") for r in d]
cols = ["0.7" if "gram" in r["kind"] else ("C0" if "γ" not in r["kind"] else "C3") for r in d]
bars(axs[0], names, [100 * r["acc"] for r in d], cols); axs[0].set_ylabel("当たり（%）")
axs[0].set_title("実験35: 次の文字の当たり（30 文字 + 〈他〉）\nカー効果を入れると学習が崩れた（失敗）", fontsize=9)
bars(axs[1], names, [min(r["bits"], 8) for r in d], cols, "{:.2f}"); axs[1].set_ylabel("交差エントロピー（ビット/文字、低いほど良い。発散は 8 で切る）")
axs[1].set_title("確率の付け方: 光（線形）は 1-gram と 2-gram の間", fontsize=9)
for ax in axs:
    ax.tick_params(axis="x", rotation=20)
save(fig, "exp35_order_colors")

# 実験36: 初期の強さと、学習後に残る形・融合
d = load(36)["rows"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=130)
sc = sorted({r["scale"] for r in d})
ch = [mm([c for r in d if r["scale"] == s for c in r["共通"]["change"]]) for s in sc]
axs[0].errorbar(sc, [c[0] for c in ch], yerr=[[c[0] - c[1] for c in ch], [c[2] - c[0] for c in ch]], fmt="o-", capsize=3)
axs[0].set_xscale("log"); axs[0].set_xlabel("初期の格子の強さ"); axs[0].set_ylabel("学習による変化 |A − A0| / |A0|")
axs[0].set_title("実験36: 初期の格子が強いほど、学習後も形が残る\n（0.3 では変化 2 割。持ち主の見立てどおり）", fontsize=9)
w = 0.25
for k, (key, lab, c) in enumerate([("共通", "共通の出発点 → 位相の平均", "C1"), ("別々", "別々の出発点 → 位相の平均", "C0")]):
    v = [mm([r[key]["fuse"]["平均"]["dist"] for r in d if r["scale"] == s]) for s in sc]
    axs[1].bar(np.arange(3) + (k - 1) * w, [x[0] for x in v], w, color=c, label=lab, yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], capsize=3)
v = [mm([r["共通"]["fuse"]["明るさを足す"]["dist"] for r in d if r["scale"] == s]) for s in sc]
axs[1].bar(np.arange(3) + w, [x[0] for x in v], w, color="C2", label="並べて明るさを足す")
axs[1].set_xticks(range(3)); axs[1].set_xticklabels([f"初期 {s}" for s in sc]); axs[1].set_ylabel("融合 1 枚の目標との差（小さいほど良い）"); axs[1].legend(fontsize=8)
axs[1].set_title("共通の出発点は位相の平均を少しだけ良くするが、\n明るさを足す（0.05）にはまだ遠い", fontsize=9)
save(fig, "exp36_shared_init")

# 実験37: 融合しやすい板
d = load(37)["rows"]
cases = list(dict.fromkeys(r["case"] for r in d))
fig, ax = plt.subplots(figsize=(9, 4), dpi=130)
v = [mm([r["fuse"] for r in d if r["case"] == c]) for c in cases]
bars(ax, cases, [x[0] for x in v], ["C0"] * (len(cases) - 1) + ["C3"], "{:.2f}", [x[1] for x in v], [x[2] for x in v])
ax.axhline(0.05, ls="--", c="C2"); ax.text(len(cases) - 0.5, 0.06, "並べて明るさを足す（0.01〜0.05）", ha="right", fontsize=8, color="C2")
ax.set_ylabel("位相を平均した 1 枚の、目標との差")
ax.set_title("実験37: 両方を知る出発点から派生させると、位相の平均 1 枚でも 2 つの意味をかなり保てる\n（2 枚が同じ「谷」にいるので、平均しても谷から出ない）", fontsize=9)
save(fig, "exp37_fusion_friendly")

# 実験38: 2 段の器で語順
d = load(38)["rows"]
try:
    d += [r for r in load("38b")["rows"] if "gram" not in r["kind"]]
except FileNotFoundError:
    pass
fig, ax = plt.subplots(figsize=(9, 4), dpi=130)
names = [r["kind"].replace("（数える）", "").replace("光の器・", "") for r in d]
bars(ax, names, [100 * r["acc"] for r in d], ["0.7" if "gram" in r["kind"] else "C3" for r in d])
ax.axhline(100 * d[1]["acc"], ls="--", c="gray", lw=0.8); ax.set_ylabel("次の文字の当たり（%）")
ax.set_title("実験38: 明るさを測って出し直す 2 段の器で、初めて 2-gram（25.9%）を越えた\n2 文字の干渉の項（組み合わせ）が、測って出し直すことで次の板に渡る", fontsize=9)
save(fig, "exp38_two_stage")

# 実験40: 圧力 + 熱だからこそ柔らかい
d = load(40)["rows"]
tools = ["圧力鍋", "プレス機", "鍋", "包丁", "ミキサー"]
fig, ax = plt.subplots(figsize=(9, 4), dpi=130)
for k, (kind, c) in enumerate(zip(dict.fromkeys(r["kind"] for r in d), ["C0", "C3"])):
    v = [mm([100 * r["soft"][t] for r in d if r["kind"] == kind]) for t in tools]
    ax.bar(np.arange(5) + (k - 0.5) * 0.4, [x[0] for x in v], 0.4, color=c, label=kind, yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], capsize=3)
ax.plot([-0.4, 0.4], [80, 80], "k--"); ax.text(0, 82, "目標 80%", ha="center", fontsize=8)
ax.set_xticks(range(5)); ax.set_xticklabels(tools); ax.set_ylabel("「柔らかい」と答えた割合（%）"); ax.legend(fontsize=8)
ax.set_title("実験40: 1 段では「圧力だけ」「熱だけ」の道具にも柔らかいが漏れる（足し算しかできない）\n2 段（明るさを測って出し直す）でプレス機の漏れが 25% → 5% に", fontsize=9)
save(fig, "exp40_combo")

# 実験41: f1 × f2 のモード
d = load(41)["rows"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), dpi=130)
modes = list(dict.fromkeys(m for r in d for m in r["per_mode"]))
for k, M in enumerate([2, 4, 8]):
    v = [mm([r["per_mode"][m]["tv"] for r in d if r["M"] == M and m in r["per_mode"]]) if any(m in r["per_mode"] for r in d if r["M"] == M) else (np.nan,) * 3 for m in modes]
    axs[0].bar(np.arange(len(modes)) + (k - 1) * 0.27, [x[0] for x in v], 0.27, label=f"モード {M} 個")
axs[0].set_xticks(range(len(modes))); axs[0].set_xticklabels(modes, fontsize=8, rotation=20); axs[0].set_ylabel("目標の分布との差"); axs[0].legend(fontsize=8)
axs[0].set_title("実験41: 1 つの器で、同じ物・同じ観点でもモードごとに答え方を変えられた（8 モードまで）", fontsize=9)
L = [mm([r["per_mode"]["ふつう"]["light"] * 100 for r in d if r["M"] == M]) for M in [2, 4, 8]]
axs[1].plot([2, 4, 8], [x[0] for x in L], "o-", c="C2"); axs[1].set_xticks([2, 4, 8]); axs[1].set_xlabel("モードの数"); axs[1].set_ylabel("「ふつう」で値に届いた光（%）")
axs[1].set_title("モードを増やすと、正確さは保つが答えに届く光が減る\n（器の覚えられる量を、明るさ = 確信度で払う）", fontsize=9)
save(fig, "exp41_modes")

# 実験42・43: 周波数の高低と量子化
d = load(42)["rows"]
conds = list(dict.fromkeys((r["part"], r["place"], r["bits"]) for r in d))
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), dpi=130)
lab = [f"{p}\n{pl}\n" + ("丸めない" if b == 0 else f"位相 {b} ビット") for p, pl, b in conds]
v = [mm([r["tv"] for r in d if (r["part"], r["place"], r["bits"]) == c]) for c in conds]
bars(axs[0], lab, [x[0] for x in v], ["C0" if c[0] == "実験42" else "C1" for c in conds], "{:.3f}", [x[1] for x in v], [x[2] for x in v])
axs[0].set_ylabel("表との差"); axs[0].tick_params(axis="x", labelsize=7)
axs[0].set_title("実験42・43: 周波数の高低はほとんど効かない。散らすと少し良い。\n観点の波は 2 ビット（4 段の位相）でもほぼ崩れない", fontsize=9)
v = [mm([100 * r["light"] for r in d if (r["part"], r["place"], r["bits"]) == c]) for c in conds]
bars(axs[1], lab, [x[0] for x in v], ["C0" if c[0] == "実験42" else "C1" for c in conds], "{:.0f}")
axs[1].set_ylabel("値に届いた光（%）"); axs[1].tick_params(axis="x", labelsize=7); axs[1].set_title("光の集まり方も、ほぼ同じ", fontsize=9)
save(fig, "exp42_band_quant")

# 実験44: 差分
d = load(44)["rows"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), dpi=130)
for ax, part, ylab, title in [(axs[0], "A", "違いとの相関（1 が最良）", "A. 違いを取り出す: 明るさの引き算はほぼ完全"),
                               (axs[1], "B", "トマトの答えとの差（0 が最良）", "B. 融合からリンゴを引いてトマトに戻す: 明るさの引き算・学習で引くが確実")]:
    keys = list(d[0][part])
    for k, (mode, c) in enumerate([("別々", "C0"), ("派生", "C1")]):
        v = [mm([r[part][key] for r in d if r["mode"] == mode]) for key in keys]
        ax.bar(np.arange(len(keys)) + (k - 0.5) * 0.4, [x[0] for x in v], 0.4, color=c, label=f"{mode}に学習した 2 枚", yerr=[[x[0] - x[1] for x in v], [x[2] - x[0] for x in v]], capsize=3)
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, fontsize=7, rotation=15); ax.set_ylabel(ylab); ax.legend(fontsize=8); ax.set_title(title, fontsize=9)
save(fig, "exp44_difference")

# 実験45・46: 初期の板の型と学習の経過
fig, axs = plt.subplots(1, 2, figsize=(13, 4.4), dpi=130)
pick = ["平ら（0）", "白色雑音 0.1", "白色雑音 ±π", "2 値（0 か π）", "縞 左ほど高い（反比例）", "縞 全部の列で 20 周", "縦の低い周波数だけの乱数（1〜8）"]
for ax, task, title in [(axs[0], "easy", "易しい課題（実験21 の表、縦 128）"), (axs[1], "hard", "重い課題（名詞 80 × 観点 9、縦 1040）")]:
    R = collections.defaultdict(list)
    for r in lines(f"46_{task}"):
        R[r["plate"]].append(r)
    for name in pick:
        c = np.array([r["curve"] for r in R[name]])
        ax.plot(c[0, :, 0], c[:, :, 1].mean(0), label=name, lw=2.2 if name in ("平ら（0）", "2 値（0 か π）") else 1.2)
    ax.set_xlabel("学習の回数"); ax.set_ylabel("表との差（3 seed の平均）"); ax.set_title(f"実験46: {title}", fontsize=9); ax.legend(fontsize=7)
    if task == "easy":
        ax.set_ylim(0, 0.5)
fig.suptitle("初期の板が「ばらばら」なほど速く覚える。平らな板は重い課題では最後まで追いつかない。2 値（0 か π）の板が安くて速い", fontsize=9)
save(fig, "exp46_plate_curves")

# 実験48: 大きなレンズ
d = lines(48)
kinds = list(dict.fromkeys(r["kind"] for r in d))
fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=130)
bars(axs[0], kinds, [np.mean([100 * r["light"] for r in d if r["kind"] == k]) for k in kinds], ["C0", "C1", "C2"], "{:.0f}%")
axs[0].set_ylabel("値に届いた光（%）"); axs[0].set_title("実験48: 大きなレンズで外の光を拾っても、値に届く光は増えない\n（拾った光は値と値の「間」にも落ちる。増えるのは読み分けられる点の数）", fontsize=9)
for k, (key, lab) in enumerate([("outside_before", "学習の前"), ("outside_after", "学習の後")]):
    axs[1].bar(np.arange(3) + (k - 0.5) * 0.4, [np.mean([100 * r[key] for r in d if r["kind"] == kk]) for kk in kinds], 0.4, label=lab)
axs[1].set_xticks(range(3)); axs[1].set_xticklabels(kinds, fontsize=8); axs[1].set_ylabel("出口で板の幅の外にある光（%）"); axs[1].legend(fontsize=8)
axs[1].set_title("これまでの器は約 3 割の光を外へ捨てていた。\n学習は、光を板の幅の中へ戻すように板を作っていた（46% → 30%）", fontsize=9)
save(fig, "exp48_big_lens")

# 実験49: 光の注意機構
d = lines(49)
fig, axs = plt.subplots(1, 3, figsize=(17, 4.4), dpi=130)
for ax, N in [(axs[0], 64), (axs[1], 256)]:
    for key, lab, c in [("address spike", "番地に並べる（スパイク）", "C3"), ("band spike", "周波数の帯に分ける（スパイク）", "C1"), ("holo spike", "1 枚に重ねる（スパイク）", "C0"),
                        ("トランスフォーマー（実数）", "比べ: トランスフォーマー", "k")]:
        Ts = sorted({r["T"] for r in d if r["N"] == N})
        v = [np.mean([r[key] for r in d if r["N"] == N and r["T"] == T and r["key"] == "でたらめ" and r["noise"] == 0 and key in r]) for T in Ts]
        ax.plot(Ts, v, "o-", c=c, label=lab, ls="--" if c == "k" else "-")
    ax.axvline(N, c="gray", lw=0.8, ls=":"); ax.text(N, 0.5, f" T = N = {N}", fontsize=8, color="gray")
    ax.set_xscale("log", base=2); ax.set_xlabel("文脈の長さ T（覚えている過去の語の数）"); ax.set_ylabel("正解率"); ax.legend(fontsize=7)
    ax.set_title(f"実験49: 鍵の置き方（N = {N}）\n1 つの場に重ねる・帯に分けると、T が N に近づくと崩れる", fontsize=9)
ax = axs[2]
Ts = sorted({r["T"] for r in d if r["N"] == 256})
for key, lab, c in [("address softmax beta=8 重み", "softmax β = 8（実験）", "C0"), ("address power p=2 重み", "明るさの 2 乗（実験）", "C1"), ("address spike 重み", "スパイク（実験）", "C3")]:
    ax.plot(Ts, [np.mean([r[key] for r in d if r["N"] == 256 and r["T"] == T and r["key"] == "でたらめ" and r["noise"] == 0]) for T in Ts], "o-", c=c, label=lab)
ax.plot(Ts, [math.exp(8) / (math.exp(8) + T - 1) for T in Ts], "k:", label="式 e^β / (e^β + T − 1)")
ax.set_xscale("log", base=2); ax.set_xlabel("文脈の長さ T"); ax.set_ylabel("正しい位置につく重み"); ax.legend(fontsize=7)
ax.set_title("softmax は無関係な語にも重み e^0 = 1 を配るので、長い文脈で薄まる\n（番地に並べる、N = 256）", fontsize=9)
save(fig, "exp49_attention")
print("ok")

# 実験50: 非線形のループで関係を何段たどっても濁らないか
d = lines(50)
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), dpi=130)
for ax, sig in zip(axs, [0.6, 1.0]):
    for kind, c in zip(["線形", "p=1", "p=2", "p=4", "スパイク"], ["k", "C0", "C1", "C2", "C3"]):
        o = np.mean([r["overlap"] for r in d if r["sigma"] == sig and r["kind"] == kind], 0)
        ax.plot(range(1, len(o) + 1), o, "o-", ms=3, c=c, label=kind.replace("p=", "明るさの ") + (" 乗" if "p=" in kind else ""))
    ax.set_xlabel("たどった段の数"); ax.set_ylabel("正しい鍵との重なり（1 = 濁りなし）"); ax.set_ylim(-0.02, 1.05); ax.legend(fontsize=8)
    ax.set_title(f"実験50: 1 段ごとに雑音（σ = {sig}）と光の減り 7% が入る鎖を 30 段たどる\n明るさを 2 乗以上して組み直すと、何段たどっても濁らない", fontsize=9)
save(fig, "exp50_chain_cleanup")
print("ok50")
