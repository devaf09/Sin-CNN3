#!/bin/bash
# 実験54 の seed 1・2（seed 0 は sweep54.sh、1 とおり 約 90 秒の実測 → 2 seed × 6 とおり ≈ 20 分）
cd "$(dirname "$0")/../.."
python experiments/exp54_kerr_focal.py --device cuda --seed 1
python experiments/exp54_kerr_focal.py --device cuda --seed 2
