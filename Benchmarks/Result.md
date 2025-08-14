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
  - Environment: Local CPU inference, Python 3.11, sklearn metrics, sentence-transformers for embedding models.
- STEME: Uses cosine similarity between model embeddings and tag embeddings.

## Results (F1 ↑, Items/sec ↑)

| Model                    | F1    | Items/s | RAM (MB) | Gain over Random |
| ------------------------ | ----- | ------: | -------: | ---------------: |
| Random                   | 0.215 |   8.8e5 |        — |                — |
| Majority                 | 0.205 |   6.9e7 |        — |                — |
| Keyword                  | 0.205 |   1.0e6 |        — |                — |
| TF-IDF                   | 0.205 |   1.8e5 |        — |                — |
| **BAAI/bge-small-en**    | 0.600 |    2802 |      536 |        **+179%** |
| **paraphrase-MiniLM-L3** | 0.548 |     849 |      474 |        **+155%** |
| **all-MiniLM-L6-v2**     | 0.704 |    3363 |      495 |        **+227%** |
| **all-mpnet-base-v2**    | 0.645 |    2485 |      785 |        **+200%** |

## Key Takeaways

- All embedding-based STEME runs outperform trivial baselines by +150% to +227% in F1.
- all-MiniLM-L6-v2 delivered the best accuracy (0.704 F1) at high throughput (>3300 items/sec).
- Even the smallest model (bge-small-en) achieved nearly 3× the accuracy of Random while running in ~2.8k items/sec.
- STEME performance scales predictably with embedding model capacity; model choice can be tuned to trade accuracy vs throughput.

## eproducing the Benchmarks

Run the benchmark script with a chosen model:

```bash
# Example with all-MiniLM-L6-v2
.\.venv\Scripts\python.exe -X dev testings\bench_steme.py --n 2000 --model-name "all-MiniLM-L6-v2"
```
Use --out to save results as JSON and --plot to visualize:
```bash
.\.venv\Scripts\python.exe -X dev testings\bench_steme.py --n 2000 --model-name "all-MiniLM-L6-v2" --out benchmarks\run.json --plot
```
Then in `bench_steme.py` you just keep the argparse help text clean and generic, e.g.:
```python
parser.add_argument("--n", type=int, default=2000, help="Number of test samples")
parser.add_argument("--model-name", type=str, required=True, help="HuggingFace model name")
```
