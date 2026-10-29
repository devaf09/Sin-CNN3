#!/bin/bash
# 実験74: 合成の語板（候補 (a)）。親の語板 v2 から学び直す（差分）とまっさらを比べる。GPU 2 枚で a・b を同時に。見込み 20〜40 分（材料の読み込みが大半）
cd "$(dirname "$0")/../.."
python experiments/exp74_compound_goban.py --device cuda --set a
