"""実験74 の結果（results/exp74.txt、非公開フォルダの compound_direction.json）から、docs/compound_74.md を作る。実行: python tools/compound_doc_74.py"""
import json

R = {}; V = None
for l in open("results/exp74.txt", encoding="utf-8"):
    d = json.loads(l)
    if "result" in d:
        R[d["set"]] = d                                                                     # 語板の当たり（集合ごとの最後の行）
    if "向き" in d:
        V = d["向き"]                                                                         # 向きの決定（最後の行 = 親自身の上位 3 に直したもの）
rows = json.load(open("D:/AI_git/Sin-CNN3_private/data/views/compound_direction.json", encoding="utf-8"))
rows.sort(key=lambda r: -abs(r["前_taken"] - r["後_taken"]))                                 # 前後の差が大きい親から
o = ["# 合成の語板（実験74）— つなげる向きと、差分で持つ合成の語板", "",
     "持ち主の案（2026-10-29）:「〜の〜 の相手が多すぎる語は、前か後ろの語をくっつけて新しい語板に。なるべく新しい語板が属性を持っていかない向きに。元の語板は残し、合成は差分で」。作り方は experiments/exp74_compound_goban.py。", "",
     f"## 1. つなげる向き（(a) 過飽和の親 {V['parents']} 語、前と後ろの両方に候補がある語）", "",
     "- **持っていく属性** = 合成語の観点（どんな・を・が・で・に・は・と）の数 ÷ 親の同じ観点の数（親の属性のうち、合成語へ移る割合）。親自身の上位 3 の相手の合計。",
     "- **親との違い** = 合成語の使われ方と親の使われ方の違い（KL、ビット）。大きいほど、別の板にする価値がある。", "",
     "| | 前につなげる（X の 親） | 後ろにつなげる（親 の Y） |", "|---|---|---|",
     f"| 選ばれた数（持っていく属性が少ない向き） | {V['front']} | {V['back']} |",
     f"| 持っていく属性（平均） | {V['taken_front'] * 100:.1f}% | {V['taken_back'] * 100:.1f}% |",
     f"| 親との違い（平均） | {V['kl_front']:.1f} ビット | {V['kl_back']:.1f} ビット |", "",
     f"- 上位 3 の相手とつないでも、親の「の前・の後」は {V['still_chosen']} / {V['parents']} 語でまだ過飽和（相手がとても多い語は、3 つでは足りない）。",
     f"- 合成語自身がまだ過飽和（さらにつなげる候補）: {V['chain_chosen']} 個。",
     f"- 「持っていかない向き」と「親との違いが小さい向き」が一致した親: {V['kl_agree']} / {V['parents']}。", "",
     "### 向きの例（前後の差が大きい親）", "", "| 親 | 選んだ向き | 前（X の 親） | 持っていく | 後ろ（親 の Y） | 持っていく |", "|---|---|---|---|---|---|"]
for r in rows[:25]:
    o.append(f"| {r['parent']} | {'前' if r['choose'] == '前' else '後ろ'} | {'・'.join(r['前'])} | {r['前_taken'] * 100:.1f}% | {'・'.join(r['後'])} | {r['後_taken'] * 100:.1f}% |")
o += ["", "## 2. 合成の語板の当たり（確かめ用の材料、交差エントロピー、ビット。小さいほど良い）", "",
      "| | (a) 過飽和 | (b) 相互情報量 |", "|---|---|---|", f"| 合成語の数（学習用の数 30 以上） | {R['a']['n']} | {R['b']['n']} |"]
for k, lab in [("parent", "親の語板そのまま（合成を作らない）"), ("diff", "親から学び直した合成の語板（差分）"), ("scratch", "まっさらから学んだ合成の語板"),
               ("table_parent", "表（親へ寄せて平ら）"), ("table_global", "表（全体へ寄せて平ら）"), ("both_parent", "両方の案に入る組: 親のまま"), ("both_diff", "両方の案に入る組: 差分の語板")]:
    o.append(f"| {lab} | {R['a']['result'][k]['bits']:.3f} | {R['b']['result'][k]['bits']:.3f} |")
o.append(f"| 板の位相が親から動いた量（平均、ラジアン） | {R['a']['result']['diff']['phase_move']:.2f} | {R['b']['result']['diff']['phase_move']:.2f} |")
for s in "ab":
    o += ["", f"### ({s}) 親より良くなった合成語（ビット、上位）", "", "・".join(f"{g[0]}（親 {g[1]}、{g[2]:+.1f}）" for g in R[s]["gain_top"][:15]),
          "", f"### ({s}) 親より悪くなった合成語", "", "・".join(f"{g[0]}（親 {g[1]}、{g[2]:+.1f}）" for g in R[s]["gain_bottom"])]
open("docs/compound_74.md", "w", encoding="utf-8").write("\n".join(o) + "\n")
