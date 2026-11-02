"""holographic.py — 構造と時の順を、1 枚の光に重ねて覚える（ホログラフィックな記憶）

「誰が・何を・いつ」のような構造と、出来事の順番を、木やリストに分けずに **1 本の光に足し合わせて** 持ちます（実験103）。
考え方は HRR・FHRR（ベクトル記号アーキテクチャ）と同じで、新しいものではありません。光の器で扱いやすい形にまとめた部品です。

    mem = HolographicMemory(dim=4096)
    mem.store(0, {"主語": "太郎", "述語": "助ける", "目的語": "亀"})
    mem.store(1, {"主語": "亀", "述語": "連れて行く", "行き先": "竜宮城"})
    mem.recall(1, "行き先")              # → ("竜宮城", 類似度)
    mem.find({"述語": "助ける"})         # → 0（その出来事がいつか）

しくみ:
    - 語・役割・時刻を、長さ D の「位相だけの光」（各点の明るさ 1、位相がでたらめ）で表す。
      位相だけの光は、位相の板そのものなので、光の装置では「板を 1 枚通す」ことに当たります。
    - 結びつけ（bind）: 点ごとの掛け算 = 位相の足し算。主語 ⊙ 太郎 は、主語とも太郎とも似ていない新しい光になる。
    - ほどく（unbind）: 逆の位相を掛ける（共役）。主語^-1 ⊙（主語 ⊙ 太郎）= 太郎。
    - 重ね合わせ（bundle）: 足し算。出来事 = Σ 役割 ⊙ 語。物語 = Σ 時刻の鍵 ⊙ 出来事。
    - 時の順: 時刻 k の鍵は τ^k（1 歩ぶんの位相を k 回まわす）。「次」は位相を 1 歩まわすだけ。
    - 取り出し: 物語 ⊙ τ^-k ⊙ 役割^-1 = 語 + でたらめな雑音。語の一覧といちばん重なるものを選ぶ（掃除、cleanup）。

容量の目安（実験103）: 必要な D は、重ねる語の数の 20〜60 倍ほど。255 語を D = 16384 で 98%、D = 4096 で 88% 取り出せた。
"""
import math

import torch


def random_phase(dim, generator=None):
    """位相がでたらめな光（各点の明るさ 1）。"""
    return torch.exp(2j * math.pi * torch.rand(dim, generator=generator)).to(torch.complex64)


def bind(a, b):
    """結びつけ: 点ごとの掛け算（位相の足し算）。"""
    return a * b


def unbind(c, a):
    """ほどく: a の逆（共役）を掛ける。bind(a, b) を a でほどくと b に戻る。"""
    return c * a.conj()


def similarity(a, b):
    """2 つの光の似ている度合い（−1〜1、向きのそろい方 = 規格化した内積の実部）。"""
    ua = a / a.norm().clamp(min=1e-12)
    ub = b / b.norm().clamp(min=1e-12)
    return float((ua.conj() * ub).sum().real)


class Codebook:
    """名前 → 位相だけの光、の辞書。初めて聞いた名前には新しい光を作る。cleanup で光から名前に戻す。"""

    def __init__(self, dim, seed=0):
        self.dim = dim
        self.g = torch.Generator().manual_seed(seed)
        self.vec = {}

    def __getitem__(self, name):
        if name not in self.vec:
            self.vec[name] = random_phase(self.dim, self.g)
        return self.vec[name]

    def cleanup(self, v, candidates=None):
        """光 v にいちばん似ている名前と、その類似度を返す（雑音を掃除して記号に戻す段）。"""
        names = list(self.vec) if candidates is None else list(candidates)
        best = max(names, key=lambda w: similarity(v, self.vec[w]))
        return best, similarity(v, self.vec[best])


class HolographicMemory:
    """出来事（役割 → 語）を時刻つきで 1 本の光に重ねて覚える。

    引数:
        dim:  光の長さ D。重ねる語の数の 20〜60 倍を目安に。
        seed: 乱数の種。
    """

    def __init__(self, dim=4096, seed=0):
        self.dim = dim
        self.words = Codebook(dim, seed)                     # 語の光
        self.roles = Codebook(dim, seed + 1)                 # 役割の光
        g = torch.Generator().manual_seed(seed + 2)
        self.step_key = random_phase(dim, g)                 # 時刻の 1 歩 τ
        self.trace = torch.zeros(dim, dtype=torch.complex64) # 物語全体（1 本の光）
        self.times = set()
        self._roles_used = set()
        self._words_used = set()

    def time_key(self, t):
        """時刻 t の鍵 τ^t（位相を t 歩まわした光）。"""
        return self.step_key ** t

    def event(self, frame):
        """出来事の光 = Σ 役割 ⊙ 語。frame は {役割: 語} の辞書。"""
        e = torch.zeros(self.dim, dtype=torch.complex64)
        for role, word in frame.items():
            e = e + bind(self.roles[role], self.words[word])
            self._roles_used.add(role)
            self._words_used.add(word)
        return e

    def store(self, t, frame):
        """時刻 t の出来事を物語に足す。"""
        self.trace = self.trace + bind(self.time_key(t), self.event(frame))
        self.times.add(t)

    def recall(self, t, role):
        """時刻 t の出来事の、役割 role の語を取り出す → (語, 類似度)。"""
        v = unbind(unbind(self.trace, self.time_key(t)), self.roles[role])
        return self.words.cleanup(v, self._words_used)

    def find(self, partial):
        """手がかり（{役割: 語} の一部）にいちばん合う出来事の時刻を返す。

        各時刻の出来事を取り出して（物語 ⊙ τ^-t）、手がかりの光と比べる。全部の時刻を同時に調べられる。
        注意: 「質問がどの出来事か」を選ぶ所は、この素朴な重なりではまだ弱い（実験103 で 12/30、実験104）。
        """
        probe = self.event(partial)
        return max(self.times, key=lambda t: similarity(unbind(self.trace, self.time_key(t)), probe))
