#!/bin/bash
# 実験77: 「隣の文に出る語」の観点の語板（材料 aozora）。GPU 2 枚で conv・aozora を同時に。見込み 5〜10 分
cd "$(dirname "$0")/../.."
python experiments/exp77_next_view.py --device cuda --source aozora
