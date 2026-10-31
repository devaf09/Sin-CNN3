#!/bin/bash
# 語板 v4（席を増やした板: 観点 9 × 値 490 + その他、器 = 縦 512 × 板 16、500 回、名詞 3 万語の半分）。持ち主 2026-10-31。見込み 1.5〜3 時間
cd "$(dirname "$0")/../.."
python experiments/exp73_goban_v2.py --device cuda --part 1 --scratch --rows 512 --plates 16 --steps 500 --nv 490 --name v4
