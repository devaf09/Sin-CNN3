#!/bin/bash
# 実験85 d: 板 32 枚・15000 回 + 真ん中の焦点面でカー効果（γ = 0.5）。見込み 1〜2 時間
cd "$(dirname "$0")/../.."
python experiments/exp85_pos_fill_synth.py --device cuda --enc freq --ref --plates 32 --steps 15000 --kerr 0.5
