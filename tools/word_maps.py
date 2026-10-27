"""単語（文字）ごとの光の 2 次元マップを保存する（持ち主の依頼 2026-10-27「適当な文章を選んで、単語ごとの 2D マップを保存してくれる？見ただけでは意味が無いけど、なにか発見があるかも」）

実験30 で学習して保存した格子（results/exp30_grid_F32.npz）を読み、文の 1 文字ずつを「1 文字後ろ」の観点（ずらし 1）で器に通し、
光が列ごとにどう進むか（縦 1024 × 列 20 の明るさ）と、出口で光った次の文字の候補を記録する。
文: 夏目漱石『吾輩は猫である』の冒頭（青空文庫、著作権切れ）。
保存: results/word_maps.npz（マップ全部、16 ビット）、docs/images/word_maps.png（全部の文字を並べた図）、docs/images/word_maps_detail.png（一部を大きく）
実行: python tools/word_maps.py
"""
import math, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
plt.rcParams["font.family"] = ["Yu Gothic", "MS Gothic", "Meiryo", "sans-serif"]
z = np.load("results/exp30_grid_F32.npz"); A = torch.tensor(z["A"]); F, N, cols, dz = int(z["F"]), int(z["N"]), int(z["cols"]), float(z["dz"])
chars = str(z["chars"]); idx = {c: i for i, c in enumerate(chars)}; S = 101; L = 2 * N
SENT = "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
pos = torch.arange(N, dtype=torch.float32); col = torch.arange(cols, dtype=torch.float32); fr = torch.arange(1, F + 1, dtype=torch.float32)
Y1, Y2 = torch.sin(2 * math.pi * fr[:, None] * pos / N), torch.cos(2 * math.pi * fr[:, None] * pos / N)
X1, X2 = torch.sin(2 * math.pi * fr[:, None] * col / cols), torch.cos(2 * math.pi * fr[:, None] * col / cols)
G = sum(Y.T @ A[k] @ X for k, (Y, X) in enumerate(((Y1, X2), (Y1, X1), (Y2, X2), (Y2, X1))))
H = torch.exp(1j * dz * torch.sqrt((2 * math.pi) ** 2 - (2 * math.pi * torch.fft.fftfreq(L)) ** 2))
freq = torch.tensor([10 * (i - S // 2) for i in range(S)]); bins = freq % N
maps, nexts = [], []
with torch.no_grad():
    for ch in SENT:
        c = idx.get(ch, 100)
        u = torch.nn.functional.pad((torch.exp(1j * 2 * math.pi * (freq[c] + 1) * pos / N) / math.sqrt(N)).to(torch.complex64), (0, L - N))
        m = []
        for k in range(cols):
            u = torch.cat([u[:N] * torch.exp(1j * G[:, k]), u[N:]])
            u = torch.fft.ifft(torch.fft.fft(u, n=L, norm="ortho") * H, n=L, norm="ortho"); m.append(u[:N].abs() ** 2)
        maps.append(torch.stack(m, 1).numpy())
        W = torch.fft.fft(u[:N], n=N, norm="ortho").abs()[bins] ** 2; W = (W / W.sum()).numpy()
        nexts.append([(chars[i] if i < 100 else "〈他〉", float(W[i])) for i in np.argsort(W)[::-1][:3]])
maps = np.array(maps)
np.savez_compressed("results/word_maps.npz", sentence=SENT, maps=maps.astype(np.float16), next3=np.array([[x[0] for x in n] for n in nexts]))
n = len(SENT); fig, ax = plt.subplots(1, n, figsize=(0.62 * n, 7), dpi=110, sharey=True)
for i, ch in enumerate(SENT):
    ax[i].imshow(np.sqrt(maps[i]), cmap="magma", aspect="auto", interpolation="nearest")
    ax[i].set_title(ch, fontsize=12); ax[i].set_xticks([]); ax[i].set_xlabel(nexts[i][0][0], fontsize=9, color="C2")
ax[0].set_ylabel("縦の位置（1024 画素）")
fig.suptitle("文字ごとの光の 2 次元マップ（観点「1 文字後ろ」、各枠 = 縦 1024 × 列 20、明るさの平方根）。下の緑の字 = 器が一番光らせた次の文字", fontsize=10)
fig.tight_layout(); fig.savefig("docs/images/word_maps.png")
pick = [0, 1, 2, 3]; fig, ax = plt.subplots(1, len(pick), figsize=(12, 6), dpi=120)
for j, i in enumerate(pick):
    ax[j].imshow(np.sqrt(maps[i][:, :]), cmap="magma", aspect="auto"); ax[j].set(title=f"「{SENT[i]}」 → 次の候補 " + "・".join(f"{c}{100 * p:.0f}%" for c, p in nexts[i]), xlabel="列（光は左から右へ）", ylabel="縦の位置"); ax[j].title.set_fontsize(8)
fig.tight_layout(); fig.savefig("docs/images/word_maps_detail.png")
for ch, nx in zip(SENT, nexts):
    print(ch, " ".join(f"{c}{100 * p:.0f}%" for c, p in nx))
