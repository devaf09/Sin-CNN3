"""sincnn3 ライブラリのテスト（リポジトリ直下で: python -m pytest tests）

どのテストも「研究で確かめた性質が、ライブラリでも成り立つか」を見ます。CPU で 30 秒ほど。
"""
import math

import pytest
import torch

from sincnn3 import (FusedGrain, HolographicMemory, KnowledgeBase, LightDevice, Not, RelationChain, bind,
                     brightness, confidence, similarity, unbind, wave)

FRUITS = {
    "リンゴ": {"色": {"赤": .8, "緑": .2}, "形": {"丸い": 1}, "味": {"甘い": .6, "酸っぱい": .4}, "触感": {"しゃきしゃき": 1}},
    "トマト": {"色": {"赤": .9, "緑": .1}, "形": {"丸い": 1}, "味": {"甘い": .3, "酸っぱい": .7}, "触感": {"むしゃむしゃ": 1}},
    "バナナ": {"色": {"黄": 1}, "形": {"細長い": 1}, "味": {"甘い": 1}, "触感": {"ねっとり": 1}},
    "イチゴ": {"色": {"赤": 1}, "形": {"丸い": .8, "細長い": .2}, "味": {"甘い": .7, "酸っぱい": .3}, "触感": {"むしゃむしゃ": 1}},
}


@pytest.fixture(scope="module")
def kb():
    k = KnowledgeBase(FRUITS, seed=0)
    k.fit(steps=300)
    return k


# ---- waves ----
def test_wave_lands_on_its_frequency():
    """周波数 f の波は、レンズの後で周波数 f に光が全部集まる。"""
    b = brightness(wave(5, 32))
    assert abs(float(b[5]) - 1) < 1e-5 and abs(float(b.sum()) - 1) < 1e-5


def test_integer_waves_are_orthogonal():
    """整数の周波数の波どうしは直交する（重ねても読み分けられる理由）。"""
    W = wave([1, 2, 7], 32)
    G = (W.conj() @ W.T).abs()
    assert torch.allclose(G, torch.eye(3), atol=1e-5)


# ---- device ----
def test_device_keeps_light_and_reverses():
    """器は光の量を保ち、逆向きに通すと元に戻る。"""
    dev = LightDevice(n=32, plates=4, seed=1)
    u = wave([3], 32)
    with torch.no_grad():
        out = dev.forward(u)
        back = dev.backward(out)
    assert abs(float((out.abs() ** 2).sum()) - 1) < 1e-4            # 場全体の光の量は 1 のまま
    assert torch.allclose(back, u, atol=1e-4)


def test_reverse_brightness_equals_forward(kb):
    """逆向きに通した明るさ = 順向きの明るさ（逆引きの土台、実験21）。"""
    fwd, back = kb.verify_reverse("トマト", "味", "酸っぱい")
    assert abs(fwd - back) < 1e-4


# ---- knowledge ----
def test_lookup_matches_table(kb):
    d = kb.lookup("リンゴ", "色")
    assert abs(d["赤"] - 0.8) < 0.1 and abs(d["緑"] - 0.2) < 0.1


def test_infer_tomato(kb):
    """「丸い・赤い・むしゃむしゃ・酸っぱい」→ トマト（実験21 の例）。"""
    ans = kb.infer({"形": "丸い", "色": "赤", "触感": "むしゃむしゃ", "味": "酸っぱい"})
    assert next(iter(ans)) == "トマト"
    assert abs(sum(ans.values()) - 1) < 1e-6
    assert 0 < confidence(ans) <= 1


def test_negation(kb):
    """「赤くない」→ 赤い物はほぼ消える（1 − p、実験53）。"""
    ans = kb.infer({"色": Not("赤")})
    assert next(iter(ans)) == "バナナ"
    assert ans["イチゴ"] < 0.05


def test_familiarity_detects_unknown(kb):
    """覚えた組み合わせはなじみが高く、ありえない組み合わせは低い（実験27・51）。"""
    assert not kb.is_unknown({"色": "黄", "形": "細長い"})
    assert kb.is_unknown({"色": "黄", "触感": "むしゃむしゃ", "形": "丸い"})


def test_fuse_is_weighted_sum(kb):
    """融合は明るさの重みつきの足し算どおり（実験34）。"""
    f = kb.fuse(["リンゴ", "バナナ"], [3, 1])
    assert isinstance(f, FusedGrain)
    want = 0.75 * kb.grains["リンゴ"].answer() + 0.25 * kb.grains["バナナ"].answer()
    assert torch.allclose(f.answer(), want, atol=1e-6)


def test_add_does_not_disturb_others(kb):
    """物を足しても、ほかの物の答えは変わらない（グレインが別々なので忘れない、実験51）。"""
    before = kb.grains["バナナ"].answer().clone()
    kb.add("サクランボ", {"色": {"赤": 1}, "味": {"甘い": .8, "酸っぱい": .2}}, start_from="イチゴ", steps=100)
    assert torch.equal(before, kb.grains["バナナ"].answer())
    assert "サクランボ" in kb.infer({"色": "赤"})


# ---- chain ----
def test_chain_cleanup_beats_linear():
    """雑音と光の減りがあっても、明るさの 2 乗できれいにすると 30 段たどれる（実験50）。"""
    names = [f"s{i}" for i in range(128)]
    acc = {}
    for p in (1, 2):
        ch = RelationChain(names, n=64, p=p, seed=0)
        for i in range(128):
            ch.relate(names[i], names[(i + 1) % 128])
        path = ch.follow("s0", hops=30, noise=1.0, loss=0.93)
        acc[p] = sum(path[h] == names[h + 1] for h in range(30)) / 30
    assert acc[2] == 1.0 and acc[1] < acc[2]


def test_chain_branches():
    ch = RelationChain(["a", "b", "c", "d"], n=64)
    ch.relate("a", "b"); ch.relate("a", "c")
    br = ch.branches("a")
    assert set(br) == {"b", "c"}


# ---- holographic ----
def test_bind_unbind_roundtrip():
    g = torch.Generator().manual_seed(0)
    a = torch.exp(2j * math.pi * torch.rand(512, generator=g))
    b = torch.exp(2j * math.pi * torch.rand(512, generator=g))
    assert similarity(unbind(bind(a, b), a), b) > 0.999
    assert abs(similarity(bind(a, b), b)) < 0.2                     # 結んだ光は元のどちらとも似ていない


def test_holographic_story():
    mem = HolographicMemory(dim=2048, seed=0)
    mem.store(0, {"主語": "太郎", "述語": "助ける", "目的語": "亀"})
    mem.store(1, {"主語": "亀", "述語": "連れて行く", "行き先": "竜宮城"})
    mem.store(2, {"主語": "太郎", "述語": "開ける", "目的語": "玉手箱"})
    assert mem.recall(1, "行き先")[0] == "竜宮城"
    assert mem.recall(0, "目的語")[0] == "亀"
    assert mem.find({"述語": "開ける"}) == 2
