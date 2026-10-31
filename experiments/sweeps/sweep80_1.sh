#!/bin/bash
# 実験80 job 1: 属性板（B 明るさだけ・C 位相込みの光）。見込み 30〜90 分
cd "$(dirname "$0")/../.."
python experiments/exp80_attribute_boards.py --device cuda --method B
python experiments/exp80_attribute_boards.py --device cuda --method C
