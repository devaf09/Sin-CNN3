#!/bin/bash
# 語板 v3（観点 9・名詞 3 万語の半分、器 = 縦 128 × 板 16、500 回。実験75 の見積もり）。GPU 2 枚で p0・p1 を同時に。見込み 1 時間（表づくりが大半）
cd "$(dirname "$0")/../.."
python experiments/exp73_goban_v2.py --device cuda --part 1 --scratch --rows 128 --plates 16 --steps 500 --name v3
