#!/bin/bash
# 実験80 job 0: 数えた表（D）と大きな器 1 枚（A、縦 2048）。見込み 30〜60 分
cd "$(dirname "$0")/../.."
python experiments/exp80_attribute_boards.py --device cuda --method D
python experiments/exp80_attribute_boards.py --device cuda --method A
