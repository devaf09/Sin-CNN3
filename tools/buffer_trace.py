"""実験89 の「上から」の答え探しで、バッファ語板がどう変わって幹（文）に着き、答えの節を選ぶかを書き出す（持ち主の依頼 2026-11-02）。
質問: 「真っ白なおじいさんは誰ですか。」（「おじいさん」の木から人物を特定する。答え = 太郎 / 浦島太郎）。実行: python tools/buffer_trace.py"""
import json, os, subprocess, sys

import torch

sys.stdout.reconfigure(encoding="utf-8")
Q = "真っ白なおじいさんは誰ですか。"
tmp = os.path.join(os.environ.get("TEMP", "."), "q_trace.txt"); open(tmp, "w", encoding="utf-8").write(Q + "\n")
out = os.path.join(os.environ.get("TEMP", "."), "q_trace.json")
subprocess.run(["D:/AI_git/Sin-CNN3_private/venv_ginza/Scripts/python", "-I", "D:/AI_git/Sin-CNN3_private/tools/parse_bunsetsu.py", tmp, out], capture_output=True)
q = json.load(open(out, encoding="utf-8"))[0]
spec = "experiments/exp89_buffer_loop.py"
src = open(spec, encoding="utf-8").read().split("R1 = run_all(Q1, A1)")[0]
ns = {"__file__": spec}; exec(compile(src, spec, "exec"), ns)
T, VOC, ROLES, cos, question, subtree_text = ns["T"], ns["VOC"], ns["ROLES"], ns["cos"], ns["question"], ns["subtree_text"]
NV = len(VOC); NR = len(ROLES)
show = lambda v, k=6: "・".join(f"{VOC[i]} {float(v[i]):.2f}" for i in torch.argsort(v[:NV], descending=True)[:k] if float(v[i]) > 0.05)
b, r = question(q)
print("質問:", Q)
print("0. バッファ（質問の手がかりの光を足したもの）の名札:", show(b), " ＋ 名詞の語板 v4 の光（長さ", round(float(b[NV + NR:].norm()), 2), "）")
print("2. 答えの役割の光（誰 → が・は・に）:", "・".join(ROLES[i] for i in range(NR) if float(r[NV + i]) > 0))
sc = sorted(((cos(next(m for m in T[i] if m["head"] == -1)["G"], b), i) for i in range(len(T))), reverse=True)
print("1. 幹（各文の根の枝葉の光）とバッファの重なり、上位 5:", "・".join(f"{i + 1} 文目 {s:.2f}" for s, i in sc[:5]))
si = sc[0][1]; nodes = T[si]
print(f"   → {si + 1} 文目「{ns['S'][si]['sentence']}」の幹へ移る")
root = next(m for m in nodes if m["head"] == -1)
print("   その幹の枝葉の光の名札:", show(root["G"], 10))
print("3. 介入: 幹の中の節ごとに、まわりの光（親の枝葉 − 自分の枝葉）がバッファをどれだけ説明するか × 役割が合うか")
best = None
for n in nodes:
    rr = float(n["g"][NV:NV + NR] @ r[NV:NV + NR]); ex = cos(n["C"], b)
    resid = (b / b.norm()) - (n["C"] / n["C"].norm().clamp(min=1e-9)) * ex                  # 差分: バッファのうち説明されずに残る分
    print(f"   「{n['text']}」 役割 {n['role']:>4}  役割の合い {rr:.0f}  説明 {ex:.2f}  残り {float(resid.norm()):.2f}  点 {rr * ex:.2f}  残りの名札: {show(resid.clamp(min=0), 3)}")
    if rr > 0 and (best is None or rr * ex > best[0]):
        best = (rr * ex, n)
print("→ 答え:", subtree_text(nodes, best[1]["id"]) if best else "（なし）")
