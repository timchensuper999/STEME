# testings/bench_steme.py
import argparse, json, time, sys, os, platform
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

# Import from your shipped STEME
from sysMod.stemeKit import embed, STEME, setModel

# Replace your gen_synth with this:
TAGS = ["sports", "technology", "finance", "entertainment", "world news"]
LEX = {
    "sports":        ["match", "coach", "league", "score", "tournament"],
    "technology":    ["startup", "software", "AI", "developer", "chip"],
    "finance":       ["market", "stocks", "bond", "inflation", "earnings"],
    "entertainment": ["film", "series", "music", "celebrity", "box office"],
    "world news":    ["summit", "diplomacy", "election", "policy", "conflict"],
}

def gen_synth(n, tags=TAGS):
    import numpy as np
    rng = np.random.default_rng(42)
    junk = ["alpha","beta","gamma","delta","omega"]
    X, y = [], []
    for _ in range(n):
        t = tags[int(rng.integers(0, len(tags)))]
        word = rng.choice(LEX[t])
        X.append(f"{word} {rng.choice(junk)} {rng.integers(0,1000)}")  # no tag token
        y.append(t)
    return X, y

def baseline_random(y_true, tags, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    preds = [tags[int(rng.integers(0, len(tags)))] for _ in y_true]
    return preds

def baseline_majority(y_true, tags):
    # For synthetic balanced data, choose first tag for determinism
    return [tags[0] for _ in y_true]

def baseline_keyword(X, tags, aliases=None):
    # aliases: dict[tag -> list[str]]; default: tokenized tag itself
    aliases = aliases or {t: [t] for t in tags}
    preds = []
    for txt in X:
        pick = None
        low = txt.lower()
        for t in tags:
            for a in aliases.get(t, [t]):
                if a.lower() in low:
                    pick = t; break
            if pick: break
        preds.append(pick)  # may be None; caller can fallback
    return preds

def baseline_tfidf(X, tags):
    # Represent texts and tags in the same TF-IDF space, cosine to tags
    # Build a doc-term space over all texts + tag strings
    corpus = X + tags
    v = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    M = v.fit_transform(corpus)                # (N+T, V)
    QT = M[:len(X)]                            # queries
    TT = M[len(X):]                            # tags
    QT = sk_normalize(QT)                      # cosine via normalized dot
    TT = sk_normalize(TT)
    S = QT @ TT.T                              # (N, T)
    idx = S.argmax(axis=1).A1                  # fastest: to flat array
    return [tags[i] for i in idx]


def measure(n, model_name, out_json, make_plot):
    process = psutil.Process(os.getpid())

    # Generate synthetic data
    X, y_true = gen_synth(n, TAGS)

        # ---------- Baselines ----------
    # 1) Random
    t0 = time.perf_counter(); preds_rand = baseline_random(y_true, TAGS); t_rand = time.perf_counter()-t0

    # 2) Majority
    t0 = time.perf_counter(); preds_maj = baseline_majority(y_true, TAGS); t_maj = time.perf_counter()-t0

    # 3) Keyword (no synonyms version: match raw tag tokens)
    t0 = time.perf_counter(); preds_kw = baseline_keyword(X, TAGS); t_kw = time.perf_counter()-t0

    # 4) TF-IDF cosine (fallback if keyword didn’t hit)
    t0 = time.perf_counter(); preds_tfidf = baseline_tfidf(X, TAGS); t_tfidf = time.perf_counter()-t0
    # If keyword returned None (no hit), replace with TF-IDF prediction
    preds_kw = [p if p is not None else q for p, q in zip(preds_kw, preds_tfidf)]

    # F1s
    f1_rand   = f1_score(y_true, preds_rand, average="micro", zero_division=0)
    f1_maj    = f1_score(y_true, preds_maj,  average="micro", zero_division=0)
    f1_kw     = f1_score(y_true, preds_kw,   average="micro", zero_division=0)
    f1_tfidf  = f1_score(y_true, preds_tfidf,average="micro", zero_division=0)


    # Warm up
    _ = embed("warmup text")

    # --- Embedding stage ---
    emb_lat_ms = []
    emb_vectors = []
    max_rss = 0

    t0_emb = time.perf_counter()
    for i, txt in enumerate(X):
        s = time.perf_counter()
        v = embed(txt)
        emb_vectors.append(v)
        emb_lat_ms.append((time.perf_counter() - s) * 1000.0)
        if i % 100 == 0:
            rss = process.memory_info().rss
            if rss > max_rss:
                max_rss = rss
    t1_emb = time.perf_counter()

    emb_thr = len(emb_vectors) / (t1_emb - t0_emb)
    emb_p50 = float(np.percentile(emb_lat_ms, 50))
    emb_p95 = float(np.percentile(emb_lat_ms, 95))

    # --- STEME stage ---
    st_lat_ms = []
    preds = []
    # --- STEME stage ---
    st_lat_ms = []
    preds = []

    def _as_label(item):
        # STEME returns (sim, item); item might be dict or string
        if isinstance(item, dict):
            for k in ("content", "text", "name", "label"):
                if k in item:
                    return str(item[k])
            return str(item)  # fallback
        return str(item)

    t0_st = time.perf_counter()
    for i, txt in enumerate(X):
        s = time.perf_counter()
        top = STEME(txt, TAGS, top_k=1)
        # top is list of (similarity, item)
        pred_label = _as_label(top[0][1])
        preds.append(pred_label)
        st_lat_ms.append((time.perf_counter() - s) * 1000.0)
        if i % 100 == 0:
            rss = process.memory_info().rss
            if rss > max_rss:
                max_rss = rss
    t1_st = time.perf_counter()


    st_thr = len(preds) / (t1_st - t0_st)
    st_p50 = float(np.percentile(st_lat_ms, 50))
    st_p95 = float(np.percentile(st_lat_ms, 95))
    f1 = f1_score(y_true, preds, average="micro")

    out = {
        "model_name": model_name,
        "n_items": int(len(preds)),
        "tag_count": int(len(TAGS)),
        "embedding": {
            "throughput_items_per_s": emb_thr,
            "latency_ms_p50": emb_p50,
            "latency_ms_p95": emb_p95,
        },
        "steme": {
            "throughput_items_per_s": st_thr,
            "latency_ms_p50": st_p50,
            "latency_ms_p95": st_p95,
        },
        "f1_micro": f1,
        "peak_rss_mb_observed": max_rss / (1024*1024),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "timestamp": time.time(),
    }

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    if make_plot:
        counts = pd.Series(preds).value_counts().reindex(TAGS, fill_value=0)
        plt.figure()
        counts.plot(kind="bar")
        plt.title("Predicted tag distribution")
        plt.ylabel("count")
        png = Path(out_json).with_suffix(".png")
        plt.tight_layout()
        plt.savefig(png, dpi=150)

    # Console summary
    print(f"{'Model':<25} | {'Emb/s':>7} | {'Emb_P50':>7} | {'Emb_P95':>7} | {'STEME/s':>7} | {'ST_P50':>7} | {'ST_P95':>7} | {'F1':>5} | {'RAM(MB)':>7}")
    print("-"*99)
    print(f"{model_name:<25} | {emb_thr:7.2f} | {emb_p50:7.3f} | {emb_p95:7.3f} | {st_thr:7.0f} | {st_p50:7.3f} | {st_p95:7.3f} | {f1:5.3f} | {max_rss/(1024*1024):7.1f}")
    print("\nBaselines")
    print("Method     |  F1   | Items/s (approx)")
    print("-------------------------------")
    print(f"Random     | {f1_rand:5.3f} | {'∞' if t_rand==0 else int(len(X)/t_rand)}")
    print(f"Majority   | {f1_maj:5.3f} | {'∞' if t_maj==0 else int(len(X)/t_maj)}")
    print(f"Keyword    | {f1_kw:5.3f} | {'∞' if t_kw==0 else int(len(X)/t_kw)}")
    print(f"TF-IDF     | {f1_tfidf:5.3f} | {int(len(X)/max(t_tfidf,1e-9))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of texts")
    ap.add_argument("--out", type=str, default="benchmarks/steme_run.json")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--model-name", type=str, default="BAAI/bge-small-en")
    args = ap.parse_args()

    setModel(args.model_name)  # Swap embedding backend here
    measure(args.n, args.model_name, args.out, args.plot)

if __name__ == "__main__":
    main()
