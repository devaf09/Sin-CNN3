#!/bin/bash
# 実験67 (b): 章板に 128・256 語、板 8・16・32 枚。見込み 2〜3 時間
cd "$(dirname "$0")/../.."
python experiments/exp67_chapter_capacity.py --device cuda --Ws 128,256 --Ks 8,16,32
