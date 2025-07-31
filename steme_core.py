from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2") # or any other preferred model

# Cache embeddings
_embed_cache = {}


def embed(text: str) -> np.ndarray:
    if text not in _embed_cache:
        _embed_cache[text] = model.encode(text, convert_to_numpy=True)
    if np.isnan(_embed_cache[text]).any() or np.linalg.norm(_embed_cache[text]) == 0:
        print(f"[Warning] Embedding degenerate for: {text}")
    return _embed_cache[text]


def STEME(input_item, pool, top_k=3):
    """
    input_item: str (query) or np.ndarray (pre-embedded vector)
    pool: list of dicts with 'content' key, or list of strings
    Returns: list of (similarity: float, memory_dict: dict) tuples
    """
    if not pool:
        return []

    # Embed the input query
    input_vec = embed(input_item) if isinstance(input_item, str) else np.array(input_item)

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
        content_text = mem.get("content")
        if not isinstance(content_text, str):
            content_text = str(content_text)
        vec = embed(content_text)
        label_vec_pairs.append((mem, vec))

    memories, vectors = zip(*label_vec_pairs)
    similarities = cosine_similarity(input_vec.reshape(1, -1), np.array(vectors))[0]
    sorted_indices = np.argsort(similarities)[::-1][:top_k]
    return [(float(similarities[i]), memories[i]) for i in sorted_indices]

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

from itertools import combinations

def STEMEnx(texts: list[str]) -> float:
    """
    Generalized version of STEME3x for n inputs.
    Computes geometric mean of all pairwise cosine similarities.
    
    texts: list of strings
    Returns a float ∈ [-1, 1] representing semantic cohesion
    """
    if len(texts) < 2:
        raise ValueError("Need at least 2 texts to compute pairwise similarity.")

    # Embed all texts
    vectors = [embed(t) for t in texts]

    # Compute all pairwise similarities
    sims = []
    for i, j in combinations(range(len(vectors)), 2):
        sim = cosine_similarity(vectors[i].reshape(1, -1), vectors[j].reshape(1, -1))[0][0]
        sims.append(sim)

    # Multiply all similarities
    product = np.prod(sims)

    # Take geometric mean
    num_pairs = len(sims)
    if product < 0:
        # Handle cube/root of negative by preserving sign
        result = -abs(product) ** (1 / num_pairs)
    else:
        result = product ** (1 / num_pairs)

    return float(result)
