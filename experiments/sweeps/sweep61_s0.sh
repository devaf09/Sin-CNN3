#!/bin/bash
# 実験61 seed 0: 会話 × 100 + ほか（Web 小説を含む、非公開・研究のみ）の混ぜ方で名詞 2 万語の語板。GPU 2 枚で seed 0・1 を同時に。見込み 30 分（実験59 の実測 9 分 + 材料の読み込み）
cd "$(dirname "$0")/../.."
python experiments/exp61_goban_mix.py --device cuda --seed 0
