#!/bin/bash
# 実験85: 品詞の穴埋めシンセ（縦 2048 × 板 16、6000 回）。見込み 1〜2 時間
cd "$(dirname "$0")/../.."
python experiments/exp85_pos_fill_synth.py --device cuda
