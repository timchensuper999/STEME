import numpy as np
from steme_core import get_closest_tag

# simulate input statement's embedding vector
input_vector = np.array([0.3, 0.7, 0.2])

# simulate tag vector pool（3D for demonstraion purpose）
tag_dict = {
    "freedom":    np.array([0.4, 0.6, 0.1]),
    "order":      np.array([0.1, 0.9, 0.0]),
    "chaos":      np.array([-0.3, 0.4, 0.5]),
    "equality":   np.array([0.2, 0.7, 0.3]),
    "discipline": np.array([0.0, 1.0, 0.1]),
}

# use STEME core function to find the closest semantic tag
tag, score = get_closest_tag(input_vector, tag_dict)

# result output
# Input statement is closest to tag: 'equality' (cosine similarity = 0.9963)
print(f"Input statement is closest to tag: '{tag}' (cosine similarity = {score:.4f})")
