#!/bin/bash
# 実験54: カー効果を場の面 / 焦点面でかけて、語順の組み合わせ（実験38 の課題）。6 とおり × 1500 回。
# GPU の番号は指定しない（共通の実行役が CUDA_VISIBLE_DEVICES を付ける）。見込み: 1 とおり 5〜15 分 × 6（実験38 の CPU の実測より速い想定）
cd "$(dirname "$0")/../.."
python experiments/exp54_kerr_focal.py --device cuda
