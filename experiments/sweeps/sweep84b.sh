#!/bin/bash
# 実験84 の直し: いつも点いている光 + 学習率を下げる + 10000 回（方式 2・4、行・句読点）。見込み 1.5〜2 時間
cd "$(dirname "$0")/../.."
for u in line punct; do for m in 2 4; do python experiments/exp84_grammar_synth.py --device cuda --unit $u --method $m --bias --steps 10000; done; done
