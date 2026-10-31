#!/bin/bash
# 実験77: 「隣の文に出る語」の観点の語板（持ち主が集めた Web 小説、実験だけ）。見込み 15〜20 分
cd "$(dirname "$0")/../.."
python experiments/exp77_next_view.py --device cuda --source web2
