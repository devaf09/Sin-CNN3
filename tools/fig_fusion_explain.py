"""図: 融合の試験の手順と、試した形（docs/fusion_explained.md 用）。
  fusion_procedure.png : 試験の流れ（2 語の板を学ぶ → 合わせる → 光を通す → 目標の平均と比べる）
  fusion_methods.png   : 試した形の一覧（板の絵と結果の数字）
実行: python tools/fig_fusion_explain.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
rng = np.random.default_rng(3)
A = rng.uniform(-1, 1, (24, 6)); B = rng.uniform(-1, 1, (24, 6))


def plate(ax, x, y, P, w=0.9, h=1.6, cmap="twilight"):
    ax.imshow(P, extent=(x, x + w, y, y + h), cmap=cmap, vmin=-1, vmax=1, aspect="auto", zorder=3)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", fc="none", ec="0.3", lw=0.8, zorder=4))


def bars(ax, x, y, p, c, w=1.2):
    for i, v in enumerate(p):
        ax.add_patch(FancyBboxPatch((x, y + i * 0.3), v * w, 0.22, boxstyle="square,pad=0", fc=c, ec="none", zorder=3))


# 1. 試験の流れ
fig, ax = plt.subplots(figsize=(16, 5.2), dpi=130); ax.set_xlim(0, 16); ax.set_ylim(0, 5.2); ax.axis("off")
steps = ["① 2 つの語の板を\n別々に学ぶ", "② 2 枚を合わせて\n「融合した 1 枚」を作る", "③ 融合した板に\n観点の光を通す", "④ 出口の明るさ\n（答えの割合）を読む", "⑤ 目標（2 語の答えの平均）\nとの差を測る"]
for i, t in enumerate(steps):
    ax.text(1.6 + i * 3.2, 4.85, t, ha="center", va="top", fontsize=10)
plate(ax, 0.5, 1.6, A); ax.text(0.95, 1.35, "リンゴの板", ha="center", fontsize=8)
plate(ax, 1.7, 1.6, B); ax.text(2.15, 1.35, "トマトの板", ha="center", fontsize=8)
ax.annotate("", (3.6, 2.4), (2.8, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5))
plate(ax, 4.3, 1.6, (A + B) / 2, cmap="cividis"); ax.text(4.75, 1.35, "融合した板\n（合わせ方はいろいろ）", ha="center", va="top", fontsize=8)
ax.annotate("", (6.8, 2.4), (5.4, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5))
xs = np.linspace(6.9, 7.6, 100)
for k in range(3):
    ax.plot(xs, 3.0 - k * 0.5 + 0.08 * np.sin(2 * np.pi * (k + 2) * (xs - 6.9) * 3), c=f"C{k}", lw=1)
ax.text(7.25, 1.5, "観点の光\n（色・味・形…）", ha="center", fontsize=8)
ax.annotate("", (9.9, 2.4), (8.6, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5))
bars(ax, 10.0, 1.4, [0.5, 0.3, 0.6, 0.2, 0.4], "C2"); ax.text(10.6, 1.1, "出口の明るさ", ha="center", fontsize=8)
ax.annotate("", (12.9, 2.4), (11.6, 2.4), arrowprops=dict(arrowstyle="->", lw=1.5))
bars(ax, 13.1, 1.4, [0.45, 0.35, 0.55, 0.25, 0.35], "0.6"); ax.text(13.7, 1.1, "目標 = (リンゴの答え + トマトの答え) / 2", ha="center", fontsize=8)
ax.text(8.0, 0.35, "差 = 全変動距離（0 = 完全に目標どおり、1 = まったく違う）。あわせて「入れた光のうち、値に届いた割合」（光の無駄）も測る。3 seed。", ha="center", fontsize=9, color="0.25")
fig.tight_layout(); fig.savefig("docs/images/fusion_procedure.png"); plt.close(fig)

# 2. 試した形
M = [("1a 並べて明るさを足す", "2 枚に光を半分ずつ通し、\n出口の明るさを足す（干渉なし）", "0.01", "34", "ok"),
     ("1b 並べて場を足す", "出口の光（場）を足す\n（干渉する）", "0.07〜0.15", "34", "mid"),
     ("2 比率 80 : 20", "明るさを 8 : 2 で足す", "0.004〜0.009", "34", "ok"),
     ("4 積み重ねる", "リンゴの板の後に\nトマトの板を続けて通す", "0.78〜0.83", "34", "ng"),
     ("5 列ごとに交互", "偶数列リンゴ・奇数列トマト", "0.79〜0.83", "34", "ng"),
     ("6 位相の平均", "板の位相を平均して 1 枚に", "0.72〜0.86", "34", "ng"),
     ("7 複素数の板", "透け具合 (e^{iA}+e^{iB})/2\n（光が減る）", "0.77〜0.92", "34", "ng"),
     ("8 学習し直す", "位相の平均から 300 回、\n両方の平均に合うよう学ぶ", "0.04〜0.06", "34", "ok"),
     ("共通の出発点", "同じ初期の板から 2 語を学び、\n位相の平均", "0.59〜0.78", "36", "mid"),
     ("両方を知る出発点から派生", "2 語の平均を 1 枚に学んでから\n各語へ 300 回、位相の平均", "0.13〜0.27", "37", "mid"),
     ("透過の模様（干渉なし）", "共通の板 + 語ごとの明るさの\nフィルム、フィルムを平均", "0.00（表す力なし）", "64", "mid")]
fig, axs = plt.subplots(3, 4, figsize=(17, 11), dpi=120)
for ax in axs.flat:
    ax.set_xlim(0, 4); ax.set_ylim(0, 3); ax.axis("off")
for ax, (name, how, err, ex, ok) in zip(axs.flat, M):
    col = {"ok": "#e3f4df", "mid": "#fff3cf", "ng": "#f6e0e0"}[ok]
    ax.add_patch(FancyBboxPatch((0.05, 0.05), 3.9, 2.9, boxstyle="round,pad=0.02", fc=col, ec="0.6"))
    ax.text(2.0, 2.75, name, ha="center", va="top", fontsize=11, fontweight="bold")
    if name.startswith("1a") or name.startswith("1b") or name.startswith("2 "):
        plate(ax, 0.3, 0.9, A, 0.45, 1.0); plate(ax, 0.3, -0.25 + 0.9 - 0.1, B, 0.45, 0.0) if False else plate(ax, 0.9, 0.9, B, 0.45, 1.0)
        ax.text(1.6, 1.4, "+", fontsize=18, va="center")
    elif name.startswith("4"):
        plate(ax, 0.3, 0.9, A, 0.45, 1.0); plate(ax, 0.78, 0.9, B, 0.45, 1.0)
    elif name.startswith("5"):
        P = A.copy(); P[:, 1::2] = B[:, 1::2]; plate(ax, 0.4, 0.9, P, 0.7, 1.0)
    elif name.startswith("6") or name.startswith("共通") or name.startswith("両方"):
        plate(ax, 0.4, 0.9, (A + B) / 2, 0.7, 1.0, cmap="cividis")
    elif name.startswith("7"):
        plate(ax, 0.4, 0.9, np.angle(np.exp(1j * A * 3) + np.exp(1j * B * 3)) / 3, 0.7, 1.0, cmap="cividis")
    elif name.startswith("8"):
        plate(ax, 0.4, 0.9, (A + B) / 2 + 0.2 * rng.uniform(-1, 1, A.shape), 0.7, 1.0, cmap="cividis")
    else:
        plate(ax, 0.4, 0.9, A, 0.45, 1.0); ax.add_patch(FancyBboxPatch((0.95, 0.9), 0.25, 1.0, boxstyle="square,pad=0", fc="0.85", ec="0.4")); plate(ax, 1.3, 0.9, B, 0.45, 1.0)
    ax.text(2.0, 1.95, how, ha="left", va="top", fontsize=9)
    ax.text(2.0, 0.7, f"目標との差: {err}", ha="left", fontsize=10, color="0.15")
    ax.text(2.0, 0.35, f"実験{ex}", ha="left", fontsize=8, color="0.4")
for ax in axs.flat[len(M):]:
    ax.text(2.0, 1.6, "緑 = うまくいった\n黄 = 一部\n赤 = 壊れた\n\n差は小さいほど良い\n（全変動距離、3 seed の範囲）", ha="center", va="center", fontsize=11)
fig.suptitle("試した融合の形（リンゴの板とトマトの板を 1 つにする）", fontsize=14)
fig.tight_layout(); fig.savefig("docs/images/fusion_methods.png"); plt.close(fig)
print("ok")
