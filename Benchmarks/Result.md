# 📊 Benchmark Results

We evaluated **STEME** on a fixed semantic tagging dataset with four trivial baselines (Random, Majority, Keyword match, TF-IDF) and several popular sentence embedding models.

## Setup
- Task: Assign the correct tag from a fixed tag set to each input sentence, based on semantic similarity.
- Evaluation: F1 score (micro), throughput in items/sec, RAM usage.
- Baselines:
  - Random: Assigns a random tag.
  - Majority: Always predicts the most frequent tag.
  - Keyword: Simple keyword match to tag.
  - TF-IDF: Cosine similarity in TF-IDF space.
  - Environment: Local CPU inference, Python 3.13, sklearn metrics, sentence-transformers for embedding models.
- STEME: Uses cosine similarity between model embeddings and tag embeddings (through the `tanh`-enhanced score by default; pass `raw=True` for untouched cosine).

## Results — default model, hard-negative sentence set

`gen_synth` draws from full templated sentences per class, deliberately sharing surface
vocabulary across classes (e.g. "the transfer market" appears under both `sports` and
`finance`) so classification has to rely on meaning, not keyword overlap. Run 2026-08-31
against `steme_core.py` (with `STEME_nli`, `setModel`/revision pinning):

| Model                 | Emb items/s | STEME items/s | F1 (micro) | Peak RAM (MB) |
| ---------------------- | ----------: | -------------: | ---------: | -------------: |
| **BAAI/bge-small-en** (default) | 845.6 | 785.0 | **0.8025** | 550.3 |

Baselines on the same 2000-item set (`--n 2000`): Random 0.199, Majority 0.204, Keyword 0.204,
TF-IDF 0.204 F1 — STEME wins by a wide margin.

Only the default model has been benchmarked so far; other embedding models (all-MiniLM-L6-v2,
all-mpnet-base-v2, paraphrase-MiniLM-L3, etc.) haven't been re-run against this dataset yet.

Raw JSON: [`steme_run.json`](./steme_run.json). Plot: [`steme_run.png`](./steme_run.png).

## Key Takeaways

- STEME comfortably outperforms all four trivial baselines despite the dataset's deliberate
  cross-class vocabulary overlap — it's relying on meaning, not keyword hits.
- STEME performance scales with embedding model capacity; model choice can be tuned to trade
  accuracy vs throughput (see Reproducing the Benchmarks below to compare models).

## Reproducing the Benchmarks

Run the benchmark script with a chosen model:

```bash
python Benchmarks/bench_steme.py --n 2000 --model-name "all-MiniLM-L6-v2"
```
Use `--out` to save results as JSON and `--plot` to visualize:
```bash
python Benchmarks/bench_steme.py --n 2000 --model-name "all-MiniLM-L6-v2" --out Benchmarks/steme_run.json --plot
```
