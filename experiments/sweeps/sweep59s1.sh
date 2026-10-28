#!/bin/bash
# 実験59 の seed 1（seed 0 は sweep59.sh）。2 枚の GPU を並べて使うため、別の仕事として入れる（持ち主の依頼 2026-10-29）。見込み 1〜3 時間。
cd "$(dirname "$0")/../.."
python experiments/exp59_goban_large.py --device cuda --seed 1
