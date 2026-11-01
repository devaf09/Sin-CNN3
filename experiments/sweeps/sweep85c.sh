#!/bin/bash
# 実験85 c: 板 32 枚・15000 回（位置ごとの周波数 + 参照の光）。カー効果なし / あり。見込み 1〜2 時間
cd "$(dirname "$0")/../.."
python experiments/exp85_pos_fill_synth.py --device cuda --enc freq --ref --plates 32 --steps 15000
