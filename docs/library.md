# sincnn3 ライブラリの説明書（版 0.1.0）

Sin-CNN3 の研究（実験01〜121）で確かめたしくみを、ほかのプロジェクトや他の人がそのまま使えるようにまとめたライブラリです。
初めての人でも読めるように、言葉の説明から書いています。しくみの紹介は [unique_results.md](unique_results.md)、数字の正本は [NOTES.md](../NOTES.md)。

## 1. 何ができるか（30 秒で）

```python
from sincnn3 import KnowledgeBase, Not

kb = KnowledgeBase({
    "リンゴ": {"色": {"赤": .8, "緑": .2}, "味": {"甘い": .6, "酸っぱい": .4}},
    "トマト": {"色": {"赤": .9, "緑": .1}, "味": {"甘い": .3, "酸っぱい": .7}},
    "バナナ": {"色": {"黄": 1},              "味": {"甘い": 1}},
})
kb.fit()                                    # 物ごとに 1 台の光の器に覚えさせる（CPU で数秒）

kb.lookup("リンゴ", "色")                   # 順引き   → {"赤": 0.80, "緑": 0.20, ...}
kb.reverse("色", "赤")                      # 逆引き   → {"トマト": 0.90, "リンゴ": 0.80, ...}
kb.infer({"色": "赤", "味": "酸っぱい"})    # 推し量り → {"トマト": 0.66, "リンゴ": 0.34, ...}（表から計算した正解も 0.66・0.34）
kb.infer({"色": Not("赤")})                 # 否定     → {"バナナ": 0.77, "リンゴ": 0.15, ...}
kb.is_unknown({"色": "黄", "味": "酸っぱい"}) # 知らない組み合わせ? → True
```

ひと通りの例は [examples/quickstart.py](../examples/quickstart.py)（CPU で 15 秒ほど）。

## 2. 入れ方

必要なもの: Python 3.9 以上と torch 2.0 以上（GPU は要りません）。

```
git clone https://github.com/devaf09/Sin-CNN3
cd Sin-CNN3
pip install -e .            # sincnn3 を使えるようにする
python examples/quickstart.py
python -m pytest tests      # テスト（pip install pytest が要る）
```

ほかのプロジェクトからは `pip install git+https://github.com/devaf09/Sin-CNN3` でも入れられます。

## 3. 言葉の説明

| 言葉 | 意味 |
|---|---|
| 波・周波数 | 記号（リンゴ・赤など）を「整数の周波数の波」で表します。整数の周波数の波どうしは直交する（混ざらない）ので、何本重ねてもあとで分けて読めます。 |
| 光の器 | 位相の板（学習する）と、板と板のあいだの回折を何枚も重ねたもの。光を増やしも減らしもしない（ユニタリ）。`LightDevice`。 |
| レンズ・明るさ | 出口でフーリエ変換して、周波数ごとの明るさ（振幅の 2 乗）を読みます。明るさの合計は 1 なので、明るさ = 割合・確率です。 |
| 観点 | 何を聞くか（色・形・味…）。入口の波の周波数で表します（観点周波数）。 |
| グレイン（語板） | 1 つの物（語）の「観点 × 値」の割合を覚えた光の器 1 台。`Grain`。 |
| 逆引き | 値から、それを持つ物を引くこと。器を逆向きに通すと、順向きと同じ明るさが出ます。 |
| なじみ | 手がかりがどれだけ覚えた道に乗るか（0〜1）。低いと「知らない組み合わせ」。 |
| 融合 | いくつかの物の知識を、出口の明るさの重みつきの足し算で混ぜること。 |
| 位相だけの光 | 各点の明るさが同じで、位相（波のずれ）だけがでたらめな光。関係のたどり・ホログラフィックな記憶で、記号の「鍵」に使います。 |

## 4. 部品の一覧

| 部品 | 何をするか | 主な関数・クラス |
|---|---|---|
| `sincnn3.knowledge` | 物と属性の表を覚え、順引き・逆引き・推し量り・否定・なじみ・融合・物を足す | `KnowledgeBase`, `Grain`, `FusedGrain`, `Not`, `confidence` |
| `sincnn3.chain` | 関係（a → b）を覚えて何段もたどる。明るさの p 乗で濁らない | `RelationChain` |
| `sincnn3.holographic` | 出来事（役割 → 語）と時の順を 1 本の光に重ねて覚える | `HolographicMemory`, `Codebook`, `bind`, `unbind`, `similarity` |
| `sincnn3.attention` | 光の注意機構。置き方（番地・帯・ホログラム）× 重み付け（softmax・べき・スパイク）を名前で選ぶ | `Attention` |
| `sincnn3.device` | 光の器そのもの。順向き・逆向きに通す | `LightDevice` |
| `sincnn3.waves` | 記号を波にする・レンズで明るさを読む | `wave`, `lens`, `brightness`, `overlap` |

各関数の細かい説明は、ソースの説明文（docstring）に書いてあります。`help(sincnn3.KnowledgeBase)` でも読めます。

## 5. 部品ごとの使い方

### 5.1 KnowledgeBase（物と属性の知識）

**表の形**: `{物: {観点: {値: 割合}}}`。割合は観点ごとに和が 1 でなくても自動でそろえます。表に書かない観点は「分からない」として学習しません。

| メソッド | 何をするか | 戻り値 |
|---|---|---|
| `fit(steps=500)` | 全部の物を学習する | `{物: 表との差}`（0 = 完全一致） |
| `lookup(物, 観点)` | 順引き | `{値: 割合}`（大きい順） |
| `reverse(観点, 値)` | 逆引き | `{物: その値を持つ割合}` |
| `infer({観点: 値 or Not(値)}, prior=None)` | 推し量り（手がかりの掛け算） | `{物: 確率}`（和 1） |
| `familiarity(手がかり)` / `is_unknown(手がかり, 0.4)` | なじみ / 知らない組み合わせか | 0〜1 / True・False |
| `fuse([物…], [重み…], name=None)` | 明るさで融合する。name を付けると知識に加わる | `FusedGrain` |
| `add(物, 表, start_from=似た物)` | 物を 1 つ足す（ほかの物は触らない） | 表との差 |
| `verify_reverse(物, 観点, 値)` | 実際に光を逆向きに通して、逆引き = 順引きを確かめる | `(順向き, 逆向き)` |

`confidence(答え)` は答えの確信度（いちばん大きい割合）。自信の高い答えだけに絞ると正解率が上がります（実験62）。

**器の大きさ**: `KnowledgeBase(表, n=64, plates=16)`。値の総数が多いときは `n ≥ 観点の数 + 2 + 値の総数` になるように n を大きくしてください（2 のべき乗が速い）。表との差を小さくしたいときは、縦（n）より板の枚数（plates）を増やすのが割に合います（実験75）。

### 5.2 RelationChain（関係をたどる）

```python
from sincnn3 import RelationChain
ch = RelationChain(["家", "駅", "電車", "会社"], n=64, p=2)
ch.relate("家", "駅"); ch.relate("駅", "電車"); ch.relate("電車", "会社")
ch.follow("家", hops=3)                            # ["駅", "電車", "会社"]
ch.follow("家", hops=3, noise=1.0, loss=0.93)       # 装置の雑音・光の減りを入れて試す
ch.branches("家")                                  # 1 段で光が届く先と明るさ
```

`p` はきれいにする強さです。p = 1 では、256 個の記号の輪を雑音つきで 30 段たどると 3 段しか当たりませんが、p = 2 では 30 段すべて当たりました（examples/quickstart.py）。

### 5.3 HolographicMemory（構造と時の順）

```python
from sincnn3 import HolographicMemory
mem = HolographicMemory(dim=4096)
mem.store(0, {"主語": "浦島太郎", "述語": "助ける", "目的語": "亀"})
mem.store(1, {"主語": "亀", "述語": "連れて行く", "行き先": "竜宮城"})
mem.recall(1, "行き先")          # ("竜宮城", 類似度)
mem.find({"述語": "助ける"})     # 0
```

`dim` は重ねる語の数の 20〜60 倍を目安にします。`find`（どの出来事かを選ぶ）は素朴な重なりで、長い物語ではまだ弱い所です（実験103・104）。

### 5.4 Attention（光の注意機構）

```python
from sincnn3 import Attention
att = Attention(store="address", select="power", p=2)
out, w = att(q, K, V)       # q: (B, N)、K, V: (T, N) の複素数
```

置き方 `store` は "address"（番地に並べる、混ざらない）・"band"（周波数の帯に分ける）・"holo"（ホログラムに重ねる）。重み付け `select` は "softmax"・"power"・"spike"。長い文脈では softmax は薄まり、power・spike は薄まりません（実験49）。

### 5.5 LightDevice と waves（土台）

```python
from sincnn3 import LightDevice, wave, brightness
dev = LightDevice(n=64, plates=8)
I = dev.brightness(wave([3, 5], 64))   # 2 本の波を通した出口の明るさ (2, 64)
u = dev.backward(dev.forward(wave(3, 64)))   # 逆向きに通すと元の波に戻る
```

自分の課題で器を学習させたいときは、`dev.phase`（torch の Parameter）を最適化します。`Grain.fit` が例です。

## 6. 研究の実験との対応

| ライブラリ | 元の実験・部品 | 違い |
|---|---|---|
| `LightDevice` | src/grain.py の器（画素ごとの板 + 回折）、実験21 の逆向き | 逆向き（backward）を足し、大きさを引数にした |
| `KnowledgeBase` | 実験21・23・24・26・27・34・51・53・62、src/grain.py | 物ごとのグレイン（語板の形、実験55〜）で作り直した。実験21 は 1 台の器に全部の物を入れていた |
| `RelationChain` | 実験17・50 | 番地に並べた記憶 + 明るさの p 乗（実験50 の形） |
| `HolographicMemory` | 実験103 | 役割と時刻の結びつけだけを取り出した（語板の位相を混ぜる部分や質問の解析は入れていない） |
| `Attention` | src/attention.py（実験49） | 同じ中身 |

研究のための実験プログラム（experiments/）と道具（tools/、テストボット02 など）は、このライブラリには含めていません。テストボット02 の部品は言結い（Project_Kotoyui）へ引き継ぎました。

## 7. 制限と注意

- すべて計算機の中の模擬（シミュレーション）です。実際の光の装置では作っていません。
- 表との差は少し残ります（果物の例で約 0.01、本物の文の 3 万語で 0.024）。数えた表そのものの方が正確です。この部品の価値は、光の装置で動く形のまま、逆引き・推し量り・否定・融合ができることです。
- `KnowledgeBase.add` で足せるのは今ある観点・値だけです。新しい観点・値を足すときは表を作り直して `fit()` してください。
- 大規模な言語モデルと同じ課題で比べたことはありません。
- 版 0.1.0 です。使いながら名前や引数が変わることがあります。変えたときは、ここと NOTES.md に書きます。
