#!/bin/bash
# 実験73: 語板 v2（観点 9・名詞 3 万語の半分）。リフレッシュ（v1 の板から）とまっさらを続けて。GPU 2 枚で p0・p1 を同時に。見込み 1〜2 時間（表づくり + 学習）
cd "$(dirname "$0")/../.."
python experiments/exp73_goban_v2.py --device cuda --part 0
python experiments/exp73_goban_v2.py --device cuda --part 0 --scratch
