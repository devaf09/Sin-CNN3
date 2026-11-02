"""sincnn3 — 波（sin・cos）と光の器で、覚える・引く・推し量るためのライブラリ

Sin-CNN3 の研究（実験01〜121）で確かめたしくみを、ほかのプロジェクトから使える形にまとめたものです。
説明書: docs/library.md、使い方の例: examples/quickstart.py、しくみの紹介: docs/unique_results.md。

部品の一覧（下の部品ほど土台）:

    knowledge   物と属性の表を光の器に覚えさせ、順引き・逆引き・推し量り・否定・なじみ・融合をする
                KnowledgeBase, Grain, FusedGrain, Not, confidence
    chain       関係（a → b）をつなげて何段もたどる。明るさの p 乗で濁らない
                RelationChain
    holographic 構造（役割 → 語）と時の順を 1 本の光に重ねて覚える
                HolographicMemory, Codebook, bind, unbind, similarity
    attention   光の注意機構（置き方 × 重み付けを名前で選んで差し替えられる）
                Attention
    device      光の器そのもの（位相の板 + 回折）。順向き・逆向きに通す
                LightDevice
    waves       記号を整数の周波数の波にする・レンズで明るさを読む
                wave, lens, brightness, overlap

すべて計算機の中の模擬（シミュレーション）で、torch だけで動きます（GPU は要りません）。
"""
from .attention import Attention
from .chain import RelationChain
from .device import LightDevice
from .holographic import Codebook, HolographicMemory, bind, random_phase, similarity, unbind
from .knowledge import FusedGrain, Grain, KnowledgeBase, Not, confidence, table_distance
from .waves import brightness, lens, overlap, wave

__version__ = "0.1.0"

__all__ = [
    "Attention", "RelationChain", "LightDevice", "Codebook", "HolographicMemory", "bind", "unbind",
    "random_phase", "similarity", "FusedGrain", "Grain", "KnowledgeBase", "Not", "confidence",
    "table_distance", "brightness", "lens", "overlap", "wave", "__version__",
]
