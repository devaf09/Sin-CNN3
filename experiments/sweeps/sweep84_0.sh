#!/bin/bash
# 実験84: 文法シンセ（単位 = line）。比べの相手と方式 1〜4。見込み 30〜60 分
cd "$(dirname "$0")/../.."
for m in base 1 2 3 4; do python experiments/exp84_grammar_synth.py --device cuda --unit line --method $m; done
