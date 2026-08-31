from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from itertools import combinations

def _enhance(raw: float, center: float = 0.38, sharpness: float = 6.0) -> float:
    # BGE-small embeddings cluster in the 0.3–0.7 cosine band; without stretching,
    # everything looks equally similar. 0.38 centers the tanh at the band midpoint.
    return float(math.tanh(sharpness * (raw - center)))

_DEFAULT_MODEL = "BAAI/bge-small-en"
_DEFAULT_REVISION = "2275a7bdee235e9b4f01fa73aa60d3311983cfea"  # pinned so embeddings don't drift if HF updates the weights
_MODEL = None
_embed_cache = {}

def setModel(model_name: str, revision: str = None):
    """Swap the active embedding model and clear the cache."""
    global _MODEL, _embed_cache
    _MODEL = SentenceTransformer(model_name, revision=revision)
    _embed_cache.clear()

def _ensure_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_DEFAULT_MODEL, revision=_DEFAULT_REVISION)

def embed(text: str, model=None) -> np.ndarray:
    if isinstance(text, np.ndarray):
        return text
    global _MODEL, _embed_cache
    if model is None:
        _ensure_model()
        m = _MODEL
    else:
        m = model
    if text not in _embed_cache:
        prompt = "Represent this sentence for retrieval: " + text
        _embed_cache[text] = m.encode([prompt], normalize_embeddings=True)[0]
    v = _embed_cache[text]
    if np.isnan(v).any() or np.linalg.norm(v) == 0:
        print(f"[Warning] Embedding degenerate for: {text}")
    return v

def STEME(input_item, pool, top_k=3, raw=False):
    """
    input_item: str (query) or np.ndarray (pre-embedded vector)
    pool: list of dicts with 'content' key, or list of strings
    raw: if True, return raw cosine similarities (no tanh enhancement) — use when
         the score is used as a magnitude/intensity value, not just for ranking.
    Returns: list of (similarity: float, memory_dict: dict) tuples
    """
    if not pool:
        return []

    # Embed the input query
    input_vec = embed(input_item) if isinstance(input_item, str) else np.array(input_item)
    try:
        input_vec = np.array(input_vec, dtype=float)  # 💥 Force float array
    except Exception as e:
        raise ValueError(f"Invalid input_vec format: {input_vec} — {e}")
    if np.isnan(input_vec).any():
        raise ValueError(f"STEME input vector contains NaNs: {input_item} → {input_vec}")

    # Normalize pool to a list of dicts with 'content'
    memory_dict_pool = []
    if isinstance(pool[0], dict) and "content" in pool[0]:
        memory_dict_pool = pool
    elif all(isinstance(p, str) for p in pool):
        memory_dict_pool = [{"content": p} for p in pool]
    else:
        raise ValueError("STEME pool must be list of strings or list of dicts with 'content'")

    # Build (dict, vector) pairs
    label_vec_pairs = []
    for mem in memory_dict_pool:
        stored = mem.get("embedding") if isinstance(mem, dict) else None
        if stored is not None:
            vec = np.array(stored, dtype=np.float32)
        else:
            content_text = mem.get("content", "") if isinstance(mem, dict) else str(mem)
            if not isinstance(content_text, str):
                content_text = str(content_text)
            vec = embed(content_text)

        if vec is None or np.isnan(vec).any():
            raise ValueError(f"Invalid vector for memory: {mem} → embedded vector = {vec}")

        label_vec_pairs.append((mem, vec))

    memories, vectors = zip(*label_vec_pairs)

    similarities = cosine_similarity(input_vec.reshape(1, -1), np.array(vectors))[0]
    scores = similarities if raw else np.array([_enhance(s) for s in similarities])
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    return [(float(scores[i]), memories[i]) for i in sorted_indices]


def STEME3x(text_a: str, text_b: str, text_c: str) -> float:
    """
    Compute the geometric mean of cosine similarities between all three pairs.
    Returns a float between -1 and 1 indicating triadic semantic cohesion.
    """
    # Embed all three inputs
    vec_a = embed(text_a)
    vec_b = embed(text_b)
    vec_c = embed(text_c)

    # Compute pairwise cosine similarities
    sim_ab = cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0]
    sim_bc = cosine_similarity(vec_b.reshape(1, -1), vec_c.reshape(1, -1))[0][0]
    sim_ac = cosine_similarity(vec_a.reshape(1, -1), vec_c.reshape(1, -1))[0][0]

    # Geometric mean of the three similarities
    product = sim_ab * sim_bc * sim_ac
    if product < 0:
        # Cube root of a negative number will be real if handled explicitly
        triadic_similarity = -abs(product) ** (1/3)
    else:
        triadic_similarity = product ** (1/3)

    return float(triadic_similarity)

def STEMEnx(texts: list[str]) -> float:
    """
    Generalized version of STEME3x for n inputs.
    Computes geometric mean of all pairwise cosine similarities.

    texts: list of strings
    Returns a float ∈ [-1, 1] representing semantic cohesion
    """
    if len(texts) < 2:
        raise ValueError("Need at least 2 texts to compute pairwise similarity.")

    vectors = [embed(t) for t in texts]

    sims = []
    for i, j in combinations(range(len(vectors)), 2):
        sim = cosine_similarity(vectors[i].reshape(1, -1), vectors[j].reshape(1, -1))[0][0]
        sims.append(sim)

    product = np.prod(sims)
    num_pairs = len(sims)
    if product < 0:
        result = -abs(product) ** (1 / num_pairs)
    else:
        result = product ** (1 / num_pairs)

    return float(result)


_EMOTION_PIPELINE = None
_EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"

# GoEmotions 28 labels → Plutchik primary category (for get_emotion_distribution compat)
GOEMOTIONS_TO_PLUTCHIK = {
    "admiration": "trust",      "amusement": "joy",          "anger": "anger",
    "annoyance": "anger",       "approval": "trust",          "caring": "trust",
    "confusion": "surprise",    "curiosity": "surprise",      "desire": "anticipation",
    "disappointment": "sadness","disapproval": "anger",       "disgust": "disgust",
    "embarrassment": "sadness", "excitement": "anticipation", "fear": "fear",
    "gratitude": "trust",       "grief": "sadness",           "joy": "joy",
    "love": "trust",            "nervousness": "fear",        "optimism": "anticipation",
    "pride": "joy",             "realization": "surprise",    "relief": "joy",
    "remorse": "sadness",       "sadness": "sadness",         "surprise": "surprise",
    "neutral": None,
}

def _ensure_emotion_model():
    global _EMOTION_PIPELINE
    if _EMOTION_PIPELINE is None:
        from transformers import pipeline as hf_pipeline
        _EMOTION_PIPELINE = hf_pipeline(
            "text-classification", model=_EMOTION_MODEL_NAME, top_k=None
        )

_EMOTION_CACHE: dict = {}

def STEME_emotion(text: str, top_k: int = 1) -> list[tuple[float, str]]:
    """
    GoEmotions classifier (roberta-base-go_emotions) for emotion detection.
    Returns (confidence, label) tuples sorted descending.
    Trained discriminatively on real emotional text — not cosine proximity to label strings.
    """
    truncated = text[:512]
    if truncated in _EMOTION_CACHE:
        return _EMOTION_CACHE[truncated][:top_k]
    with _MODEL_LOCK:
        if truncated not in _EMOTION_CACHE:
            _ensure_emotion_model()
            results = _EMOTION_PIPELINE(truncated)
            sorted_results = sorted(results[0], key=lambda x: x["score"], reverse=True)
            _EMOTION_CACHE[truncated] = [(float(r["score"]), r["label"]) for r in sorted_results]
    return _EMOTION_CACHE[truncated][:top_k]


_NLI_PIPELINE = None
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
_NLI_CACHE: dict = {}

import threading
_MODEL_LOCK = threading.Lock()

def _ensure_nli():
    global _NLI_PIPELINE
    if _NLI_PIPELINE is None:
        from transformers import pipeline
        _NLI_PIPELINE = pipeline("zero-shot-classification", model=_NLI_MODEL_NAME)

def STEME_nli(text: str, labels: list[str], top_k: int = 1, multi_label: bool = False) -> list[tuple[float, dict]]:
    """
    Zero-shot NLI classification against a fixed label set.
    Returns (confidence, {"content": label}) tuples, sorted descending.

    Use for classifying text into a predefined concept space (beliefs, social roles).
    Unlike cosine STEME, NLI understands implication — not just vocabulary overlap.
    "I told him the truth even when it hurt" → "honesty": 0.82, not 0.45 cosine noise.

    args:
        text:        The sentence to classify.
        labels:      Candidate label strings (e.g. meta_tag_pool["belief"]).
        top_k:       How many top results to return.
        multi_label: If True, scores are independent per-label (sigmoid, not softmax).
                     Use when comparing against existing cluster names so scores don't
                     collapse as the number of labels grows.
    """
    cache_key = (text, tuple(sorted(labels)), multi_label)
    if cache_key in _NLI_CACHE:
        paired = _NLI_CACHE[cache_key]
    else:
        with _MODEL_LOCK:
            if cache_key not in _NLI_CACHE:  # re-check after acquiring lock
                _ensure_nli()
                result = _NLI_PIPELINE(text, candidate_labels=labels, multi_label=multi_label)
                paired = sorted(zip(result["scores"], result["labels"]), reverse=True)
                _NLI_CACHE[cache_key] = paired
            paired = _NLI_CACHE[cache_key]
    return [(float(score), {"content": label}) for score, label in paired[:top_k]]


def inverse_embed(embedding: np.ndarray, top_k: int = 3) -> list[tuple[float, str]]:
    """
    Nearest-neighbour lookup against the embed cache.
    Returns the top_k cached strings geometrically closest to the given embedding.
    No perfect inverse exists — treats this as a semantic diagnostic.

    Args:
        embedding:    The vector to invert.
        top_k:        How many candidates to return.
        filter_keys:  Optional explicit list of cache keys to restrict search to.
                      Useful for e.g. only searching among emotion labels or belief labels.

    Returns:
        List of (similarity, string) tuples, descending by similarity.

    Example:
        inverse_embed(agent.memory_core.emotion_vector)
        # → [(0.91, "anxiety"), (0.87, "fear"), ...]

        inverse_embed(belief_cluster["centroid"], filter_keys=meta_tag_pool["belief"])
        # → [(0.88, "order"), (0.85, "conformity"), ...]
    """
    if not _embed_cache:
        return []

    keys = list(_embed_cache.keys())
    if not keys:
        return []

    vectors = np.array([_embed_cache[k] for k in keys])
    sims = cosine_similarity(embedding.reshape(1, -1), vectors)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]

    return [(float(sims[i]), keys[i]) for i in top_indices]


def STEME_trait(content: str, trait_data: dict) -> float:
    """
    Hybrid Scoring: Recycles STEME for batch similarity.
    Uses Power Scaling (sim^2) to naturally suppress noise
    without needing a hard threshold.
    """
    # 1. Base Similarity to the abstract definition
    base_res = STEME(content, [trait_data["defi"]], top_k=1)
    base_sim = base_res[0][0] if base_res else 0.0

    # 2. Behavioral Sensor Match (Positives & Negatives)
    pos_res = STEME(content, trait_data["pos"], top_k=2) if trait_data.get("pos") else []
    neg_res = STEME(content, trait_data["neg"], top_k=2) if trait_data.get("neg") else []

    top_pos = sum(r[0] for r in pos_res) / 2 if pos_res else 0.0
    top_neg = sum(r[0] for r in neg_res) / 2 if neg_res else 0.0

    # 3. Hybrid Signal Calculation
    polarity = 1.0 if top_pos >= top_neg else -1.0

    # 4. Power Scaling to suppress noise
    raw_signal = (0.25 * top_pos) - (0.25 * top_neg) + (0.5 * polarity * base_sim)

    # Square the magnitude but keep the sign
    trait_signal = np.sign(raw_signal) * (abs(raw_signal) ** 2)

    return float(np.clip(trait_signal, -1.0, 1.0))
