#!/bin/bash
# 実験78: 品詞の観点（D 案）。数え直し（CPU、数分）+ 語板 5 万語 × 観点 8（縦 64 × 板 16、500 回）。見込み 20〜30 分
cd "$(dirname "$0")/../.."
python experiments/exp78_pos_views.py --device cuda
