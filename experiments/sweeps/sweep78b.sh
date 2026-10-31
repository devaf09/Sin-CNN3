#!/bin/bash
# 実験78 のやり直し（出口のずらしを入れた）+ 実験79 の job 0（観点 9・18、板 16）。見込み 30〜40 分
cd "$(dirname "$0")/../.."
rm -f results/.exp78b_done
python experiments/exp78_pos_views.py --device cuda && touch results/.exp78b_done
python experiments/exp79_18views.py --device cuda --job 0
