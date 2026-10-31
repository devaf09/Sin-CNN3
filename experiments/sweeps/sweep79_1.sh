#!/bin/bash
# 実験79 job 1（観点 18・板 24・32）。品詞の語板（実験78 のやり直し）ができるまで待ってから。見込み 30〜40 分
cd "$(dirname "$0")/../.."
sleep 60
until [ -f results/.exp78b_done ]; do sleep 30; done
python experiments/exp79_18views.py --device cuda --job 1
