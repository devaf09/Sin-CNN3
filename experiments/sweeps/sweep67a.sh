#!/bin/bash
# 実験67 (a): 章板に 32・64 語、板 8・16・32 枚 + 単位板の模型。GPU 2 枚で a・b を同時に。見込み 1〜2 時間（実験66 の CPU 実測から）
cd "$(dirname "$0")/../.."
python experiments/exp67_chapter_capacity.py --device cuda --Ws 32,64 --Ks 8,16,32 --unit
