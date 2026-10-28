"""実験51: 知らない物に出会ったら、新しいグレイン（学習器）を自動で作る（持ち主の案、2026-10-28）

着想: 持ち主「赤ちゃんが物を覚えるみたいに、既存のグレインと当てはまる量が少ない場合、自動で学習器を 1 つ作って、それに属性を当てはめる機構を。
      例: リンゴとトマトを学習済み、なすびを初めて知る。属性がどれとも合わない率が高い（少しでも合わなければ別の物、95% とか）と、
      新しいなすびのグレインを作る。普通に新しい学習器を作り、周囲の属性から学びを始める。この学習をある種別の分別グレインと直交させたい」。
      これは適応共鳴理論（ART、Grossberg 1976〜）の「警戒（vigilance）のしきい値を下回ったら新しい分類を作る」と同じ形。
方法: 部品 src/grain.py（1 物 = 1 グレイン = 画素の板 8 列、観点 5・値 14）。
  A. 合う率としきい値: 学習済みのグレイン 4 つ（リンゴ・トマト・イチゴ・キュウリ）。観察（文から読んだつもりの「観点 → 値」）6 つ
     （リンゴ・青リンゴ・トマト・ナス・バナナ・ピーマン）。合う率 = 観察した観点ごとの明るさ（割合）の幾何平均。
     しきい値 ρ より合う率が低ければ「知らない物」→ 新しいグレイン。ρ を変えて、どれが新しい物と判定されるか。
  B. 新しいグレインの学び始め: ナスの観察は一部の観点だけ（3 観点 = 色 紫・食べ方 加熱・分類 野菜 / 4 観点 = それに 形 長い）。作り方 3 つ:
       まっさら（2 値の板から）/ いちばん近いグレインを写す（合う率が最大のもの）/ 近いグレインを写してから、観察した 3 観点だけ 100 回学習（周囲の属性から学び始める）
     見ていない観点を、本当のナスの表とどれくらい当てられるか（受け継ぎ）。そのあと本当の表を全部見せて学習したときの速さ。
  C. 直交（ほかを壊さない）: 1 枚の共有の板（物ごとに周波数をずらす、実験21 の形）にナスを足して、ナスだけで学習し直すと、ほかの物を忘れるか。
     グレインを別々にすれば、ほかのグレインは 1 画素も変わらない。
指標: 合う率、表との差（全変動距離）、学習の速さ（表との差 0.15 を切った回）。3 seed。結果の置き場: results/exp51.txt
実行: CUDA_VISIBLE_DEVICES= python experiments/exp51_new_grain.py
"""
import json, math, os, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from grain import Grain, tv

sys.stdout.reconfigure(encoding="utf-8"); torch.set_num_threads(2)
VIEWS = {"色": ["赤", "緑", "紫", "黄"], "形": ["丸い", "長い"], "味": ["甘い", "酸っぱい", "薄い", "苦い"], "食べ方": ["生", "加熱"], "分類": ["果物", "野菜"]}
VAL = [x for v in VIEWS.values() for x in v]; VN = list(VIEWS)
TABLE = {
    "リンゴ": {"色": {"赤": .7, "緑": .2, "黄": .1}, "形": {"丸い": 1}, "味": {"甘い": .8, "酸っぱい": .2}, "食べ方": {"生": .9, "加熱": .1}, "分類": {"果物": 1}},
    "トマト": {"色": {"赤": .9, "緑": .1}, "形": {"丸い": 1}, "味": {"酸っぱい": .6, "甘い": .4}, "食べ方": {"生": .6, "加熱": .4}, "分類": {"野菜": .7, "果物": .3}},
    "イチゴ": {"色": {"赤": 1}, "形": {"丸い": .8, "長い": .2}, "味": {"甘い": .6, "酸っぱい": .4}, "食べ方": {"生": 1}, "分類": {"果物": 1}},
    "キュウリ": {"色": {"緑": 1}, "形": {"長い": 1}, "味": {"薄い": .8, "苦い": .2}, "食べ方": {"生": .8, "加熱": .2}, "分類": {"野菜": 1}},
    "ナス": {"色": {"紫": 1}, "形": {"長い": .8, "丸い": .2}, "味": {"薄い": .9, "苦い": .1}, "食べ方": {"加熱": .9, "生": .1}, "分類": {"野菜": 1}},
    "バナナ": {"色": {"黄": 1}, "形": {"長い": 1}, "味": {"甘い": 1}, "食べ方": {"生": .9, "加熱": .1}, "分類": {"果物": 1}},
    "ピーマン": {"色": {"緑": 1}, "形": {"丸い": .6, "長い": .4}, "味": {"苦い": .8, "薄い": .2}, "食べ方": {"加熱": .8, "生": .2}, "分類": {"野菜": 1}},
}
KNOWN = ["リンゴ", "トマト", "イチゴ", "キュウリ"]
OBS = {"リンゴ": {"色": "赤", "形": "丸い", "味": "甘い", "食べ方": "生", "分類": "果物"},
       "青リンゴ": {"色": "緑", "形": "丸い", "味": "酸っぱい", "食べ方": "生", "分類": "果物"},
       "トマト": {"色": "赤", "形": "丸い", "味": "酸っぱい", "食べ方": "加熱", "分類": "野菜"},
       "ナス": {"色": "紫", "形": "長い", "味": "薄い", "食べ方": "加熱", "分類": "野菜"},
       "バナナ": {"色": "黄", "形": "長い", "味": "甘い", "食べ方": "生", "分類": "果物"},
       "ピーマン": {"色": "緑", "形": "丸い", "味": "苦い", "食べ方": "加熱", "分類": "野菜"}}


def target(obj):
    t = torch.zeros(len(VN), len(VAL))
    for i, v in enumerate(VN):
        for x, p in TABLE[obj][v].items():
            t[i, VAL.index(x)] = p
    return t


def match(g, obs):
    """合う率: 観察した観点ごとに、観察した値の割合（明るさを観点内で割合に直す）の幾何平均。"""
    W = g.answer(); r = []
    for v, x in obs.items():
        i = VN.index(v); cols = [VAL.index(y) for y in VIEWS[v]]
        r.append(float(W[i, VAL.index(x)] / W[i, cols].sum()))
    return math.exp(sum(math.log(max(x, 1e-4)) for x in r) / len(r))


out = "results/exp51.txt"
for seed in range(3):
    G = {o: Grain(len(VN), len(VAL), seed=seed + 10 * k) for k, o in enumerate(KNOWN)}
    for o in KNOWN:
        G[o].fit(target(o), steps=600)
    # A. 合う率
    M = {name: {o: match(G[o], obs) for o in KNOWN} for name, obs in OBS.items()}
    for name in OBS:
        best = max(M[name], key=M[name].get)
        print(seed, f"A {name}: いちばん合う {best} {M[name][best]:.3f}  " + " ".join(f"{o} {M[name][o]:.3f}" for o in KNOWN), flush=True)
    # B. ナスのグレインの学び始め（文から読めた観点だけ見る。3 観点 / 4 観点の 2 通り）
    tN = target("ナス"); B = {}
    for case, seen in [("3 観点", {"色": "紫", "食べ方": "加熱", "分類": "野菜"}), ("4 観点", {"色": "紫", "形": "長い", "食べ方": "加熱", "分類": "野菜"})]:
        mask = torch.tensor([v in seen for v in VN]); obs_t = torch.zeros_like(tN)
        for v, x in seen.items():
            obs_t[VN.index(v), VAL.index(x)] = 1.0
        near = max(KNOWN, key=lambda o: match(G[o], {k: x for k, x in seen.items() if k != "色"}))   # 紫はどれも持たないので、色以外で近さを測る
        for kind in ["まっさら", "近いグレインを写す", "写して見た観点だけ学ぶ"]:
            g = Grain(len(VN), len(VAL), seed=seed + 99) if kind == "まっさら" else G[near].copy()
            if kind != "近いグレインを写す":
                g.fit(obs_t, steps=100, mask=mask)
            guess = tv(g.answer(), tN, ~mask)                                    # 見ていない観点の当たり具合（受け継ぎ）
            curve = g.fit(tN, steps=400)                                         # 本当の表を全部見せて学習
            hit = next((25 * i for i, c in enumerate(curve) if c < 0.15), None)
            B[f"{case} {kind}"] = dict(near=near, unseen_tv=guess, steps_to_015=hit, final=curve[-1], curve=curve)
            print(seed, f"B {case} {kind}（近い = {near}）: 見ていない観点の表との差 {guess:.3f}  全部見せて 0.15 を切った回 {hit}  最後 {curve[-1]:.3f}", flush=True)
    # C. 直交: 共有の板に足すと忘れるか
    objs = KNOWN + ["ナス"]
    waves = [6 * k + 1 + i for k in range(len(objs)) for i in range(len(VN))]   # 物 k・観点 i の入口の周波数（桁を分ける置き方）
    S = Grain(len(waves), len(VAL), seed=seed, waves=waves, val0=40)
    tS = torch.cat([target(o) for o in objs]); old = torch.arange(len(waves)) < 5 * len(KNOWN)
    S.fit(tS, steps=1500, mask=old)
    before = tv(S.answer(), tS, old)
    S2 = S.copy(); S2.fit(tS, steps=300, mask=~old)                             # ナスだけで学習し直す
    after_only = tv(S2.answer(), tS, old); new_only = tv(S2.answer(), tS, ~old)
    S3 = S.copy(); S3.fit(tS, steps=300)                                        # 全部をもう一度見せる（古いデータも要る）
    after_all = tv(S3.answer(), tS, old); new_all = tv(S3.answer(), tS, ~old)
    sep_old = sum(tv(G[o].answer(), target(o)) for o in KNOWN) / len(KNOWN)
    C = dict(shared_before=before, shared_newonly_old=after_only, shared_newonly_new=new_only, shared_all_old=after_all, shared_all_new=new_all, separate_old=sep_old)
    print(seed, "C 共有の板: 古い物の表との差 " + f"{before:.3f} → ナスだけ学習 {after_only:.3f}（ナス {new_only:.3f}）/ 全部学習 {after_all:.3f}（ナス {new_all:.3f}）  別々のグレイン: 古い物 {sep_old:.3f}（変わらない）", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(exp="51", seed=seed, A=M, B=B, C=C), ensure_ascii=False) + "\n")
