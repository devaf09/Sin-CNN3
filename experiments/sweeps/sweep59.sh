#!/bin/bash
# 実験59: 語板を数万語に（名詞 2 万・動詞 3000、2000 回、5000 枚ずつ）。材料は非公開フォルダ（D:/AI_git/Sin-CNN3_private）。
# GPU の番号は指定しない（共通の実行役が CUDA_VISIBLE_DEVICES を付ける）。見込み: 1〜3 時間（CPU の実測 400 枚 × 2000 回 ≈ 2〜10 分から、GPU で 10 倍速い想定）
cd "$(dirname "$0")/../.."
python experiments/exp59_goban_large.py --device cuda
