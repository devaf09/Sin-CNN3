#!/bin/bash
# 実験69: テストボット01 用の語板（名詞 5000・動詞 1000、会話向けの値）。材料の読み込みに 10〜20 分 + 学習 数分。非公開フォルダに保存。
cd "$(dirname "$0")/../.."
python experiments/exp69_testbot_model.py --device cuda
