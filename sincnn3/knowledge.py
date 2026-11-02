"""knowledge.py — 物と属性の知識を光の器に覚えさせ、手がかりから推し量る

いちばん使う部品です。「リンゴの色は 赤 80%・緑 20%」のような表を渡すと、物ごとに 1 台の光の器
（グレイン = 語板。「1 語 = 1 枚の器」、実験31・55）に覚えさせ、次のことができます。

    kb = KnowledgeBase({"リンゴ": {"色": {"赤": .8, "緑": .2}, "味": {"甘い": .6, "酸っぱい": .4}}, ...})
    kb.fit()                                            # 学習（物ごとに 1〜2 秒、CPU）
    kb.lookup("リンゴ", "色")                           # 順引き → {"赤": 0.80, "緑": 0.20}
    kb.reverse("色", "赤")                              # 逆引き → {"トマト": 0.9, "リンゴ": 0.8, ...}
    kb.infer({"色": "赤", "味": "酸っぱい"})            # 推し量り → {"トマト": 0.66, "リンゴ": 0.34, ...}
    kb.infer({"色": Not("赤")})                         # 否定 → 「赤くない」物
    kb.familiarity({"色": "赤", "形": "細長い"})        # なじみ（低いと「知らない組み合わせ」）
    kb.fuse(["リンゴ", "トマト"], [0.8, 0.2])           # 明るさで混ぜた新しい知識

しくみ（docs/unique_results.md の 2〜5・7 章）:
    1. 観点周波数: 観点 v（色・形…）は入口の波の周波数 v で表す。物の器に周波数 v の波を入れると、
       その観点の答えの値 j が、出口の周波数 val0 + j に明るく出る。明るさの割合 = 値の割合。
    2. 逆引き: 器はユニタリなので、出口（値）から逆に通した明るさ = 順向きの明るさ（device.py）。
       だから「赤を持つ物」は、物ごとの「色 → 赤」の明るさを並べるだけで分かる。
       （verify_reverse で、実際に逆向きに通して等しいことを確かめられます。）
    3. 推し量り: 手がかりごとの明るさを掛け算する = 確率の掛け算（素朴ベイズ）。合わない物は 0 に近づく。
    4. 否定: 「赤くない」は、赤に行かなかった残りの光 1 − p（実験53）。
    5. なじみ: 手がかりの明るさの幾何平均。覚えた道に乗らない組み合わせは暗い（驚き、実験27・51）。
    6. 融合: 板の位相は混ぜず、出口の **明るさ** を足す（実験34・64）。

制限（正直に）:
    - 1 つの物に入れられる値の数は n − val0 個まで（既定 n=64 なら観点 9 のとき約 50 個）。
    - 表との差は少し残ります（果物の例で 約 0.01、本物の文の 3 万語で 0.024、実験55〜58・75）。数えた表そのものの方が正確です。
      この部品の価値は、光の装置で動く形のまま、逆引き・推し量り・否定・融合ができることです。
"""
import math

import torch

from .device import LightDevice
from .waves import wave


class Not:
    """否定の手がかり。kb.infer({"色": Not("赤")}) で「赤くない」を表す（明るさ 1 − p を使う）。"""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Not({self.value!r})"


# ---------------------------------------------------------------------------
# グレイン: 1 つの物の知識を持つ、1 台の光の器
# ---------------------------------------------------------------------------
class Grain:
    """1 つの物（語）の「観点 × 値」の割合を覚える光の器。

    引数:
        n_views: 観点の数 V。観点 v（0 始まり）は周波数 v + 1 の波で問い合わせる。
        n_vals:  値の総数（全部の観点の値を 1 列に並べた数）。値 j は出口の周波数 val0 + j で読む。
        n, plates, gap, seed: 光の器の大きさ（LightDevice と同じ）。板は既定 16 枚（8 枚だと表との差が 5 倍ほど大きい）。
        device:  既にある器を使う（copy から作るとき）。

    答えの読み方:
        answer() は (観点の数, 値の数) の明るさの表。行 v は観点 v を問い合わせたとき各値に来た光の量。
        行の合計は 1 より少し小さい（残りの光は値以外の出口へ行った分）。
    """

    def __init__(self, n_views, n_vals, n=64, plates=16, gap=64.0, seed=0, device=None):
        self.n_views, self.n_vals = n_views, n_vals
        self.val0 = n_views + 2                                     # 値の周波数は観点の周波数より上に置く（混ざらない）
        if device is None:
            device = LightDevice(n=n, plates=plates, gap=gap, seed=seed)
        self.device = device
        if self.val0 + n_vals > device.n:
            raise ValueError(f"値が多すぎます: 観点 {n_views}・値 {n_vals} には n ≥ {self.val0 + n_vals} が要ります（今 n = {device.n}）")
        self.inputs = wave([v + 1 for v in range(n_views)], device.n)  # 観点の波 (V, n)

    def _bright(self):
        return self.device.brightness(self.inputs)[:, self.val0:self.val0 + self.n_vals]

    def answer(self):
        """(観点の数, 値の数) の明るさの表を返す（学習しない計算）。"""
        with torch.no_grad():
            return self._bright()

    def fit(self, target, mask=None, steps=500, lr=0.01):
        """target（観点 × 値の割合の表、行ごとに和 1）に合わせて板を学習する。

        引数:
            target: (観点の数, 値の数) のテンソル。その観点で出ない値は 0。
            mask:   学習に使う観点（True の行だけ）。表に無い観点は False にする。
            steps, lr: 学習の回数と速さ（Adam）。既定 500 回（実験75 で十分だった回数）。果物の例（板 16 枚）で表との差 約 0.01。
        戻り値:
            学習後の「表との差」（全変動距離、0 = 完全に一致、1 = まったく違う）。

        学び方: 出口の明るさ（＋「値以外へ行った残りの光」）を確率と見て、表との交差エントロピーを下げる。
        """
        m = torch.ones(self.n_views, dtype=torch.bool) if mask is None else mask
        opt = torch.optim.Adam([self.device.phase], lr=lr)
        q = torch.cat([target, torch.zeros(self.n_views, 1)], 1)
        for _ in range(steps):
            W = self._bright()
            p = torch.cat([W, (1 - W.sum(1)).clamp(min=1e-9)[:, None]], 1)   # 残りの光も 1 つの「その他」として数える
            loss = -(q[m] * torch.log(p[m] + 1e-9)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return table_distance(self.answer(), target, m)

    def copy(self):
        """板を写した新しいグレイン（似た物から学び始めると速い・見ていない属性も受け継ぐ、実験51）。"""
        return Grain(self.n_views, self.n_vals, device=self.device.copy())


class FusedGrain:
    """いくつかのグレインの明るさを重みつきで足したもの（融合、実験34・64）。

    板の位相を混ぜると壊れるが、出口の明るさは足しても干渉しない。だから融合は完全に足し算どおりになる。
    """

    def __init__(self, grains, weights):
        s = float(sum(weights))
        self.grains, self.weights = list(grains), [w / s for w in weights]
        self.n_views, self.n_vals = grains[0].n_views, grains[0].n_vals

    def answer(self):
        return sum(w * g.answer() for g, w in zip(self.grains, self.weights))


def table_distance(W, target, mask=None):
    """明るさの表 W と割合の表 target の差（全変動距離。観点ごとに割合に直して比べ、平均）。"""
    m = torch.ones(len(target), dtype=torch.bool) if mask is None else mask
    Wn = W / W.sum(1, keepdim=True).clamp(min=1e-12)
    return float((Wn - target).abs().sum(1)[m].mean() / 2)


# ---------------------------------------------------------------------------
# 知識の集まり: 名前で使えるようにした入口
# ---------------------------------------------------------------------------
class KnowledgeBase:
    """物と属性の表から、物ごとのグレインを作って、引く・逆引きする・推し量る。

    引数:
        table: {物: {観点: {値: 割合}}} の辞書。割合は観点ごとに和が 1 でなくても構いません（自動でそろえます）。
               表に無い観点は「分からない」として学習しません。
        n, plates, gap: 光の器の大きさ。値の総数が多いときは n を大きく（n ≥ 観点の数 + 2 + 値の総数）。
        seed:  乱数の種（同じ種なら同じ結果）。

    例は examples/quickstart.py。
    """

    def __init__(self, table, n=64, plates=16, gap=64.0, seed=0):
        self.table = {}
        self.views, self.values = [], []                         # 観点の名前の一覧・値の名前の一覧（観点をまたいで 1 列）
        self._value_view = {}                                    # 値 → それが属する観点の一覧
        self.n, self.plates, self.gap, self.seed = n, plates, gap, seed
        self.grains = {}
        for obj, views in table.items():
            self._register(obj, views)

    # ---- 表の登録 ----
    def _register(self, obj, views):
        self.table[obj] = views
        for v, vals in views.items():
            if v not in self.views:
                self.views.append(v)
            for val in vals:
                if val not in self.values:
                    self.values.append(val)
                self._value_view.setdefault(val, set()).add(v)

    def _target(self, obj):
        """物の表を (観点 × 値) のテンソルと、学習に使う観点の印に直す。"""
        T = torch.zeros(len(self.views), len(self.values))
        mask = torch.zeros(len(self.views), dtype=torch.bool)
        for v, vals in self.table[obj].items():
            i = self.views.index(v)
            s = sum(vals.values())
            for val, p in vals.items():
                T[i, self.values.index(val)] = p / s
            mask[i] = True
        return T, mask

    def _new_grain(self, k):
        return Grain(len(self.views), len(self.values), n=self.n, plates=self.plates, gap=self.gap, seed=self.seed + k)

    # ---- 学習 ----
    def fit(self, steps=500, lr=0.01, verbose=False):
        """全部の物のグレインを学習する。戻り値: {物: 表との差}。

        注意: 観点や値の種類が増えると器の形が変わるので、学習は全部やり直しになります。
              あとから物を 1 つ足すときは add() を使ってください（ほかの物は壊しません）。
        """
        errs = {}
        for k, obj in enumerate(self.table):
            T, mask = self._target(obj)
            g = self._new_grain(k)
            errs[obj] = g.fit(T, mask, steps=steps, lr=lr)
            self.grains[obj] = g
            if verbose:
                print(f"  {obj}: 表との差 {errs[obj]:.3f}")
        return errs

    def add(self, obj, views, start_from=None, steps=500, lr=0.01):
        """物を 1 つ足して、その物のグレインだけ学習する（ほかのグレインは触らないので忘れない、実験51）。

        引数:
            start_from: 似ている既存の物の名前。その板を写して学び始めると速く、表に書かなかった観点も受け継ぐ。
        制限: 今ある観点・値だけを使えます（新しい観点・値を足すと器の形が変わるため。そのときは fit() をやり直す）。
        """
        for v, vals in views.items():
            if v not in self.views or any(val not in self.values for val in vals):
                raise ValueError("add() では今ある観点・値だけ使えます。新しい観点・値を足すときは表を作り直して fit() してください")
        self._register(obj, views)
        g = self.grains[start_from].copy() if start_from else self._new_grain(len(self.grains))
        T, mask = self._target(obj)
        err = g.fit(T, mask, steps=steps, lr=lr)
        self.grains[obj] = g
        return err

    def fuse(self, objs, weights, name=None):
        """いくつかの物の知識を、明るさの重みつきの足し算で混ぜる（例: リンゴ 80% + トマト 20%）。

        name を渡すと、その名前で知識に加わり、lookup・infer の対象になります。戻り値: FusedGrain。
        """
        f = FusedGrain([self.grains[o] for o in objs], weights)
        if name is not None:
            self.grains[name] = f
            self.table[name] = {}
        return f

    # ---- 読む ----
    def _dist(self, obj, view):
        """物 obj の観点 view の値の割合（その観点の値だけで和 1 にそろえる）。テンソル (値の数,)。"""
        W = self.grains[obj].answer()[self.views.index(view)]
        ok = torch.tensor([view in self._value_view[val] for val in self.values])
        W = W * ok
        return W / W.sum().clamp(min=1e-12)

    def lookup(self, obj, view, top=None):
        """順引き: 物の観点を聞く → {値: 割合}（大きい順）。"""
        d = self._dist(obj, view)
        out = {self.values[j]: float(d[j]) for j in range(len(self.values)) if view in self._value_view[self.values[j]]}
        return _top(out, top)

    def reverse(self, view, value, top=None):
        """逆引き: 「この観点でこの値を持つ物は？」→ {物: その物がその値を持つ割合}（大きい順）。

        光の装置では、値の光を出口から逆向きに通して入口の明るさを読みます。器はユニタリなので
        その明るさは順向きの明るさと等しく（verify_reverse で確かめられます）、ここでは順向きの明るさを使います。
        """
        j = self.values.index(value)
        out = {o: float(self._dist(o, view)[j]) for o in self.grains}
        return _top(out, top)

    def infer(self, clues, prior=None, top=None):
        """推し量り: 手がかりから、どの物らしいか → {物: 確率}（大きい順、和 1）。

        引数:
            clues: {観点: 値} の辞書。値の代わりに Not(値) を渡すと否定（1 − p）。
                   書かなかった観点は「分からない」として何も掛けない。
            prior: {物: 事前の重み}（よく出る物ほど大きく、など）。省略すると全部同じ。
        しくみ: 物ごとに、手がかりの明るさ（否定なら 1 − 明るさ）を掛け合わせ、全体で 1 にそろえる。
        """
        scores = {}
        for o in self.grains:
            s = 1.0 if prior is None else float(prior.get(o, 0.0))
            for view, val in clues.items():
                if isinstance(val, Not):
                    s *= 1.0 - float(self._dist(o, view)[self.values.index(val.value)])
                else:
                    s *= float(self._dist(o, view)[self.values.index(val)])
            scores[o] = s
        z = sum(scores.values())
        out = {o: (s / z if z > 0 else 0.0) for o, s in scores.items()}
        return _top(out, top)

    def familiarity(self, clues, obj=None):
        """なじみ: 手がかりがどれだけ「覚えた道」に乗るか（0〜1）。

        物ごとに手がかりの明るさの幾何平均をとり、obj を省略したらいちばん高い物の値を返す。
        しきい値 0.4 を下回ったら「知らない組み合わせ」と見るのが目安（実験51）。
        知らない使われ方の文は、覚えた文より 75〜340 倍暗くなった（実験27）。
        """
        def g(o):
            ps = []
            for view, val in clues.items():
                p = float(self._dist(o, view)[self.values.index(val.value if isinstance(val, Not) else val)])
                ps.append(1 - p if isinstance(val, Not) else p)
            return math.exp(sum(math.log(max(p, 1e-12)) for p in ps) / max(len(ps), 1))
        if obj is not None:
            return g(obj)
        return max(g(o) for o in self.grains)

    def is_unknown(self, clues, threshold=0.4):
        """なじみがしきい値より低ければ True（「知らない物・知らない使われ方」かもしれない）。"""
        return self.familiarity(clues) < threshold

    def verify_reverse(self, obj, view, value):
        """逆引きの明るさを、実際に光を逆向きに通して計算し、順向きの明るさと並べて返す。

        戻り値: (順向きの明るさ, 逆向きの明るさ)。2 つは計算の誤差（1e-6 ほど）を除いて等しい。
        しくみ: 出口で値 j の周波数の波を作り、器を逆に通して、入口で観点 v の周波数の明るさを読む。
        """
        g = self.grains[obj]
        if isinstance(g, FusedGrain):
            raise ValueError("融合した知識は明るさの足し算なので、逆向きには通せません")
        v, j = self.views.index(view), self.values.index(value)
        dev = g.device
        fwd = float(g.answer()[v, j])
        with torch.no_grad():
            # 出口でレンズの後に周波数 val0 + j に立つ光 = レンズの前では、その周波数の波
            u_out = wave(g.val0 + j, dev.n)[None]
            u_in = dev.backward(u_out)
            back = float((torch.fft.fft(u_in, dim=-1, norm="ortho").abs() ** 2)[0, v + 1])
        return fwd, back


def confidence(dist):
    """答えの確信度: いちばん大きい割合（0〜1）。光が 1 つの出口に集まるほど高い。

    自信の高い答えだけに絞ると正解率が上がる（13% → 43%、実験62）。
    dist は {名前: 割合} の辞書（infer や lookup の戻り値）。
    """
    return max(dist.values()) if dist else 0.0


def _top(d, top):
    items = sorted(d.items(), key=lambda kv: -kv[1])
    return dict(items[:top] if top else items)
