#!/bin/bash
# 実験46: 共通の板の型 15 種 × 3 seed、易しい課題（実験21 の表）と重い課題（名詞 80 × 観点 9）。途中から再開できる（終わった型・seed は飛ばす）。
# GPU の番号は指定しない（共通の実行役が CUDA_VISIBLE_DEVICES を付ける）。見込み: 重い課題 45 秒 × 45 本 ≈ 35 分 + 易しい課題 ≈ 10 分（2026-10-28 の試しの実測）
cd "$(dirname "$0")/../.."
python experiments/exp46_plate_patterns.py --task easy --device cuda
python experiments/exp46_plate_patterns.py --task hard --device cuda
