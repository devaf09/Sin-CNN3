#!/bin/bash
# 実験75: 器を大きくする（job 1）。GPU 2 枚で 0・1 を同時に。見込み 1〜2 時間
cd "$(dirname "$0")/../.."
python experiments/exp75_bigger_device.py --device cuda --job 1
