import numpy as np
import pytest
from steme_core import embed, STEME, STEME3x, STEMEnx

# ---------- helpers ----------
def _toy_pool_str():
    # 5 coarse tags; strings are enough for STEME() pool
    return ["sports", "technology", "finance", "entertainment", "world news"]

def _toy_pool_dict():
    return [{"content": s} for s in _toy_pool_str()]

def _make_texts_for(tag: str, n=5, seed=0):
    rng = np.random.default_rng(seed)
    junk = ["alpha", "beta", "gamma", "delta", "omega"]
    return [f"{tag} {rng.choice(junk)} {i}" for i in range(n)]

# ---------- tests: embed ----------
def test_embed_returns_unit_norm_vector():
    v = embed("hello world")
    assert isinstance(v, np.ndarray)
    n = np.linalg.norm(v)
    assert np.isfinite(n)
    assert abs(n - 1.0) < 1e-3  # normalized

def test_embed_deterministic_same_input_same_vector():
    a = embed("same text")
    b = embed("same text")
    assert np.allclose(a, b, atol=1e-6)

def test_embed_handles_unicode():
    u = embed("早安，世界")
    assert np.isfinite(u).all()
    assert abs(np.linalg.norm(u) - 1.0) < 1e-3

# ---------- tests: STEME core ----------
def test_steme_topk_ordering_str_pool():
    pool = _toy_pool_str()
    res = STEME("sports event highlights", pool, top_k=3)
    # returns list[(similarity, item)], sorted descending
    assert isinstance(res, list) and 1 <= len(res) <= 3
    sims = [r[0] for r in res]
    assert all(isinstance(s, float) for s in sims)
    assert all(sims[i] >= sims[i+1] for i in range(len(sims)-1))

def test_steme_equivalent_str_vs_dict_pool():
    pool_s = _toy_pool_str()
    pool_d = _toy_pool_dict()
    q = "latest technology trends"
    s = STEME(q, pool_s, top_k=3)
    d = STEME(q, pool_d, top_k=3)
    # compare the 'content' field for dicts with the strings
    # compare by content regardless of representation (str or dict)
    def _as_content(x):
        v = x[1]  # (similarity, item)
        if isinstance(v, dict):
            # prefer common keys; fall back to stringify
            for k in ("content", "text", "name"):
                if k in v:
                    return str(v[k])
        return str(v)

    s_items = [_as_content(x) for x in s]
    d_items = [_as_content(x) for x in d]
    assert s_items == d_items


def test_steme_accepts_preembedded_input_vector():
    pool = _toy_pool_str()
    qv = embed("finance market news")
    res = STEME(qv, pool, top_k=1)  # pass vector directly
    assert isinstance(res, list) and len(res) == 1

def test_steme_empty_pool_returns_empty():
    assert STEME("anything", [], top_k=3) == []

def test_steme_invalid_pool_raises():
    with pytest.raises(ValueError):
        STEME("x", [123, {"no_content": "y"}], top_k=1)

def test_steme_deterministic_repeatability():
    pool = _toy_pool_str()
    q = "entertainment gossip"
    a = STEME(q, pool, top_k=3)
    b = STEME(q, pool, top_k=3)
    assert a == b

def test_steme_topk_clipped_when_pool_small():
    pool = ["sports"]
    r = STEME("sports news", pool, top_k=5)
    assert len(r) == 1

def test_steme_vector_dtype_float64_ok():
    pool = _toy_pool_str()
    v = embed("world news today").astype(np.float64)
    r = STEME(v, pool, top_k=1)
    assert len(r) == 1

def test_steme_handles_nonstring_content_by_str_coercion():
    pool = [{"content": 42}, {"content": "sports"}]
    r = STEME("sports", pool, top_k=2)
    # Just smoke-test: should not raise; sorts by similarity
    assert len(r) == 2

# ---------- tests: n-wise cohesion ----------
def test_steme3x_symmetry_and_bounds():
    a, b, c = "sports", "technology", "finance"
    v1 = STEME3x(a, b, c)
    v2 = STEME3x(b, a, c)
    assert np.isfinite(v1) and -1.0 <= v1 <= 1.0
    assert abs(v1 - v2) < 1e-8

def test_stemenx_matches_3x_for_n3():
    texts = ["sports", "technology", "finance"]
    assert abs(STEMEnx(texts) - STEME3x(*texts)) < 1e-8

def test_stemenx_requires_at_least_two():
    with pytest.raises(ValueError):
        STEMEnx(["only one"])
