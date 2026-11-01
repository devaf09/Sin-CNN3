#!/bin/bash
# 実験85 のやり直し: 参照の光あり（位置 = 位相）と、位置ごとに別の周波数（参照あり）。見込み 20〜40 分
cd "$(dirname "$0")/../.."
python experiments/exp85_pos_fill_synth.py --device cuda --enc phase --ref
python experiments/exp85_pos_fill_synth.py --device cuda --enc freq --ref
