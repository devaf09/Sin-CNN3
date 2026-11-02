"""chain.py — 関係をつなげてたどる（何段たどっても濁らない）

「A → B」「B → C」… のような関係を覚え、出口の光をそのまま次の入口に入れて何段もたどります（実験17）。
装置が不完全で毎段に雑音や光の減りが入っても、**明るさを p 乗して組み直す**と濁りがたまりません（実験50）。

    ch = RelationChain(["家", "駅", "電車", "会社"], n=64)
    ch.relate("家", "駅"); ch.relate("駅", "電車"); ch.relate("電車", "会社")
    ch.follow("家", hops=3)                  # → ["駅", "電車", "会社"]
    ch.follow("家", hops=3, noise=1.0)       # 雑音が大きくても、p=2 のきれいにする段で正しくたどれる

しくみ:
    - 記号ごとに「位相がでたらめな光」（長さ n、明るさはどこも同じ）を鍵として持つ。
      でたらめな光どうしはほぼ直交する（重なり 約 1/n）。
    - 関係 a → b は、鍵 a の番地に、値として鍵 b を置く（番地に並べた記憶、attention.py の "address"）。
    - 1 段: 問いの光 q と全部の鍵の重なりの明るさ I を測る → 重み w = I^p / Σ I^p → 次の問い = Σ w · 値。
    - p = 1 だと外れの明るさ 1/n がそのまま次へ混ざり、段を重ねると濁る。
      p = 2 以上だと外れは 1/n² 以下に縮み、毎段ほぼきれいな鍵から出発し直せる（デジタルの「0 か 1 に戻す」と同じ役）。
      光の部品では、明るさを測って出し直す、または可飽和吸収体のような「明るいほど通す」材料に当たります。
    - 枝分かれ（a → b と a → c）があると、光は両方へ分かれて同時に進みます（branches() で見られます）。
"""
import math

import torch

from .attention import Attention


class RelationChain:
    """記号のあいだの関係（a → b）を覚えて、何段もたどる。

    引数:
        symbols: 記号の名前の一覧。
        n:       鍵の光の長さ。記号の数より十分大きいほど、鍵どうしの混ざりが小さい（目安 n ≥ 記号の数）。
        p:       きれいにする強さ（明るさの p 乗）。1 = きれいにしない、2〜4 = 実験50 で濁らなかった値。
        seed:    乱数の種。
    """

    def __init__(self, symbols, n=64, p=2, seed=0):
        self.symbols = list(symbols)
        self.n, self.p = n, p
        self.g = torch.Generator().manual_seed(seed)
        # 位相がでたらめな光（各画素の明るさ 1/n、合計 1）
        self.keys = torch.exp(2j * math.pi * torch.rand(len(self.symbols), n, generator=self.g)) / math.sqrt(n)
        self.pairs = []                                            # 覚えた関係 (a の番号, b の番号)

    def relate(self, a, b):
        """関係 a → b を覚える。"""
        self.pairs.append((self.symbols.index(a), self.symbols.index(b)))

    def _memory(self):
        """番地に並べた記憶: 鍵 = 関係の出発点の光、値 = 行き先の光。"""
        K = torch.stack([self.keys[i] for i, _ in self.pairs])
        V = torch.stack([self.keys[j] for _, j in self.pairs])
        return K, V

    def step(self, q, noise=0.0, loss=1.0):
        """1 段たどる。q: 問いの光 (n,)。戻り値: 次の問いの光 (n,)（長さ 1 に戻してある）。

        noise: 装置の位相の雑音の強さ（σ ラジアン）。loss: 1 段で残る光の割合（実験17 では 0.93）。
        """
        q = q * torch.exp(1j * noise * torch.randn(self.n, generator=self.g)) * math.sqrt(loss)
        K, V = self._memory()
        att = Attention("address", "power", p=self.p)
        out, _ = att(q[None], K, V)
        out = out[0]
        return out / out.norm().clamp(min=1e-12)                   # 利得: 明るさを 1 に戻す

    def nearest(self, q):
        """光 q にいちばんよく重なる記号の名前と、その重なりの明るさ（0〜1）。"""
        I = (self.keys.conj() @ q).abs() ** 2
        i = int(I.argmax())
        return self.symbols[i], float(I[i] / (q.abs() ** 2).sum())

    def follow(self, start, hops, noise=0.0, loss=1.0):
        """start から hops 段たどり、各段でいちばん明るい記号の名前のリストを返す。"""
        q = self.keys[self.symbols.index(start)].clone()
        path = []
        for _ in range(hops):
            q = self.step(q, noise, loss)
            path.append(self.nearest(q)[0])
        return path

    def branches(self, start, threshold=0.1):
        """1 段たどったとき、光が分かれて届く記号 → 明るさの割合（枝分かれを同時に見る）。"""
        q = self.keys[self.symbols.index(start)].clone()
        K, V = self._memory()
        out, _ = Attention("address", "power", p=1)(q[None], K, V)
        I = (self.keys.conj() @ out[0]).abs() ** 2
        I = I / I.sum()
        return {s: float(x) for s, x in sorted(zip(self.symbols, I.tolist()), key=lambda t: -t[1]) if x >= threshold}
