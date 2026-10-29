"""実験65: 知性樹（介入の樹形図）— 共通の属性で語板を絞り込み、確信度の高いものを融合し、親子の差を「属性差グレイン」として残す

着想: 2026-10-29 持ち主「同じ属性を持つ語板から、属性の不確実性が低い板の上位数割を抽出する。その語板から関連性の高い属性を探す（抽出した板を融合して 1 回で）。
      これを何度か繰り返すと、最終的に残った板の属性は拡散する。その一つ前に戻って何が違うのかを調べる。一つずつ戻って、ルートまで戻ったら 2 番目の属性で同じことを。
      属性グレイン → 語板の対応ツリー。ツリーの親子は結構似ているはず。知性樹に。例: 語板 1000 枚 → 共通する属性が一番多い語板集（A 集）→ 確信度の高い上位 1 割（B 集）→
      その中で次に共通する属性で抽出（C1 集）→ 関連性を属性差グレインとして作成 → 1 つ戻って B 集の 2 番目の属性（C2 集）→ … 共通する属性がほぼなくなるまで」。
方法: 実験58 の決まりで、会話の材料（非公開）から名詞 1200 語の語板を作る（平らにした目標・出口のずらし・2000 回）。属性 = (観点, 値)（「その他」は除く）。
  節（語板の集まり S）ごとに:
    融合した板 = S の語板の出口の明るさの平均（学び 34: 明るさで足す融合は確実）。
    語板 n が属性 a を「持つ」= 出口の割合 ≥ max(0.05, 全体の割合の 3 倍)。共通の度合い(a) = S のうち a を持つ語板の割合。
    共通の度合いが大きい順に、まだ使っていない属性で枝を作る（最大 3 本）。枝の集まり = a を持つ語板のうち、a の割合（確信度）が高い上位 5 割（最低 3 枚。3 割では 2 段で 3 枚まで減って木が浅かった）。
    子の融合した板 − 親の融合した板 = 属性差グレイン（どの属性が強まり・弱まったか）。
    共通の度合いが 0.05 未満・集まりが 3 枚未満・深さ 4 で止める（属性が拡散した = 葉）。（最初は 0.3 にしたが、根の 1200 語ではどの属性も 3 割に届かず、木が育たなかった）
結果の置き場: results/exp65.txt（木の数字）、docs/knowledge_tree_65.md（木の読みやすい形）
実行: CUDA_VISIBLE_DEVICES= python experiments/exp65_knowledge_tree.py
"""
import json, os, runpy, sys, time

import torch

sys.argv = [sys.argv[0], "--seeds", "0"]
ns = runpy.run_path(os.path.join(os.path.dirname(__file__), "exp56_goban_grow.py"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from goban_batch import train, run, prob

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(4); t00 = time.time()
NTR, NOUNS, NVALS, NVIEW, prior = ns["NTR"], ns["NOUNS"], ns["NVALS"], ns["NVIEW"], ns["prior"]
STEP = 9


def shift(T):
    out = T.clone()
    for v in range(T.shape[1]):
        out[:, v, [(r + STEP * v) % 47 for r in range(47)]] = T[:, v, :47]
    return out


T = shift(NTR); pr = prior(T); tg = (T + 5 * pr[None]) / (T.sum(2, keepdim=True) + 5)
P = train(tg, torch.ones(tg.shape[:2], dtype=torch.bool), 2000, 0, lr=0.03); Pr = prob(run(P, len(NVIEW), tg.shape[2]))
print("語板", len(NOUNS), f"{time.time() - t00:.0f} 秒", flush=True)
ATTR = [(j, (r + STEP * j) % 47, f"{v} {NVALS[v][r]}") for j, v in enumerate(NVIEW) for r in range(47)]   # (観点, 出口, 名前)
has = torch.stack([Pr[:, j, x] >= max(0.05, 3 * float(pr[j, x])) for j, x, _ in ATTR], 1)        # (語, 属性) 持つか
strength = torch.stack([Pr[:, j, x] for j, x, _ in ATTR], 1)                                     # (語, 属性) 割合 = 確信度
fused = lambda S: torch.stack([Pr[S, j, x].mean() for j, x, _ in ATTR])                           # 融合した板（属性ごとの割合）
nodes = []; KEEP = float(os.environ.get("KEEP", "0.5")); MIN_SHARE = 0.05                                                      # 共通の度合いがこれ未満なら「属性が拡散した」として止める


def grow(S, used, depth, path):
    i = len(nodes); f = fused(S)
    nodes.append(dict(id=i, depth=depth, path=path, size=len(S), words=[NOUNS[k] for k in S[:8]], children=[]))
    share = has[S].float().mean(0)
    for a in used:
        share[a] = -1
    order = torch.argsort(share, descending=True)
    branches = 0
    for a in order.tolist():
        if branches >= 3 or depth >= 4 or share[a] < MIN_SHARE:
            break
        members = [k for k in S if has[k, a]]
        members = sorted(members, key=lambda k: -float(strength[k, a]))
        keep = members[:max(3, int(KEEP * len(members)))]
        if len(keep) < 3 or len(keep) == len(S):
            continue
        fc = fused(keep); d = fc - f
        up = [(ATTR[t][2], round(float(d[t]), 3)) for t in torch.argsort(d, descending=True)[:4].tolist()]
        down = [(ATTR[t][2], round(float(d[t]), 3)) for t in torch.argsort(d)[:3].tolist()]
        j = grow(keep, used + [a], depth + 1, path + [ATTR[a][2]])
        nodes[j].update(attr=ATTR[a][2], share=round(float(share[a]), 3), diff_up=up, diff_down=down, parent=i)
        nodes[i]["children"].append(j); branches += 1
    if not nodes[i]["children"]:
        top = torch.argsort(f, descending=True)[:5].tolist()
        nodes[i]["leaf_attrs"] = [(ATTR[t][2], round(float(f[t]), 3)) for t in top]
        nodes[i]["max_share_left"] = round(float(share.max()), 3)                                 # 共通の度合いの残り（小さい = 属性が拡散した）
    return i


grow(list(range(len(NOUNS))), [], 0, [])
print("節の数", len(nodes), f"{time.time() - t00:.0f} 秒", flush=True)
with open("results/exp65.txt", "a", encoding="utf-8") as fo:
    fo.write(json.dumps(dict(exp="65", nouns=len(NOUNS), nodes=nodes, sec=time.time() - t00), ensure_ascii=False) + "\n")
lines = ["# 知性樹（実験65）— 共通の属性で絞り込み、親子の差を属性差グレインとして残した木", "",
         "実験65（experiments/exp65_knowledge_tree.py）で自動で作った。会話の材料の名詞 1200 語の語板から。",
         "各節: 枝の属性（親の中でその属性を持つ割合）・語板の数・語の例。「強まった / 弱まった」= 子の融合した板 − 親の融合した板（属性差グレイン）。", ""]


def show(i, indent):
    n = nodes[i]
    if i == 0:
        lines.append(f"- **根**（{n['size']} 語）")
    else:
        lines.append(f"{'  ' * indent}- **{n['attr']}**（親の {int(100 * n['share'])}% が持つ → 確信度の高い {n['size']} 語）: {'・'.join(n['words'])}")
        lines.append(f"{'  ' * indent}  - 強まった: {', '.join(f'{a} +{v:.2f}' for a, v in n['diff_up'])} ／ 弱まった: {', '.join(f'{a} {v:.2f}' for a, v in n['diff_down'])}")
        if "leaf_attrs" in n:
            lines.append(f"{'  ' * indent}  - 葉（共通の度合いの残り {int(100 * n['max_share_left'])}%）: {', '.join(f'{a} {v:.2f}' for a, v in n['leaf_attrs'])}")
    for c in n["children"]:
        show(c, indent + 1)


show(0, 0)
open("docs/knowledge_tree_65.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines[:60]))
