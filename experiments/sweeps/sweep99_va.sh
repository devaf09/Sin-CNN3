#!/bin/bash
# 実験99: 語板 v5 の動詞 3000・形容詞と形状詞 2000（観点 13・10）。見込み 30〜60 分
cd "$(dirname "$0")/../.."
python experiments/exp99_goban_v5.py --device cuda --group verb
python experiments/exp99_goban_v5.py --device cuda --group adj
