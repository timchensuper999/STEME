# STEME

**“The shortest distance between your statement and tag is STEME.”**

---

## Semantic Tagging Embedding Model Engine

STEME is a lightweight, deterministic module for **semantic tag assignment** using embedding vector similarity.  
It works by locating where your statement lives in vector space—and identifying which tag it’s closest to.

Unlike large language models (LLMs), STEME does **not** rely on context windows, prompt templates, or grammatical parsing.  
It’s small, fast, and powerful: a true utility layer for meaning.

---

## Why STEME?

STEME was built on a simple idea:

> **Every expression, when embedded, carries a semantic direction.**  
> **STEME tells you where it points.**

It takes:
- An input string or embedding (a sentence, a belief, a reflection…)
- A pool of tags or content to compare against (emotions, ideologies, memory logs…)

It returns:
- The most semantically aligned item(s), using cosine similarity.

---

## What Makes It Special?

STEME is not a model. It’s a **projection engine**.

It can work with:
- Individual sentences
- Memory clusters (vector-averaged)
- Cross-lingual inputs
- Visual captions (via compatible embedding)
- Any vector-based conceptual content

It can map them to:
- Emotional categories
- Philosophical stances
- Cognitive frames
- Abstract ideals like “justice”, “hope”, “order”, or “freedom”

---

## 🔧 Key APIs

```python
embed(text: str) → np.ndarray
STEME(query: str or vector, pool: list[str or dict], top_k=3) → [(score, item)]
STEME3x(text_a, text_b, text_c) → float  # Triadic cohesion
STEMEnx(list_of_texts) → float           # N-way cohesion
```
All methods return cosine-based similarity between -1 and 1 (usually in [0, 1] for well-formed embeddings).

---

## 🚀 Example Usage
```python
from steme_core import STEME, STEMEnx

texts = ["I love music", "Sound makes me happy", "Silence is golden"]
query = "I enjoy listening to tunes"

print("Top matches:")
for sim, match in STEME(query, texts):
    print(f"{sim:.3f} → {match['content']}")

print("\nSemantic cohesion of list:")
print(STEMEnx(texts))
```

---

## 🧠 Use Cases
- Civilizism Simulation Engine: Semantic memory tagging for AI social agents
- Belief Conflict Detection: Identify incompatible ideas in cognitive systems
- Sentiment & Emotion Modeling: Tagging emotional tone or psychological state
- Educational NLP: Assessing student writing by concept drift and tone
- Sapir-Whorf Research: Testing cross-language semantic anchoring

---

## 🗺️ Future Plans
- Add support for alternate embedding backends (OpenAI, Claude, etc.)
- Expose low-level vector utilities and cluster monitoring
- Package as plugin for cognitive simulation platforms
- Long-term: paper on belief coherence + Sapir-Whorf applications

---

## ⛓️ Installation
STEME requires the following:
```python
pip install sentence-transformers scikit-learn numpy
```

---

## 📜 License
MIT License – open, permissive, and integration-friendly.\
Use it, fork it, plug it into your brain.

---

## 🙌 Credits
Created by Tim Chen\
Originally designed for the [Civilizism Project]{https://github.com/timchensuper999/Civilizism}\
Now released as a standalone toolkit for cognition-oriented NLP.

---

> “Tags are Coordinates, Not Definitions.”
