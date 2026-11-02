#!/bin/bash
# 実験99: 語板 v5 の名詞 2 万語（観点 17 = 意味 9 + 品詞 8、縦 256 × 板 24）。見込み 1〜1.5 時間
cd "$(dirname "$0")/../.."
python experiments/exp99_goban_v5.py --device cuda --group noun
