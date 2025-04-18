from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_closest_tag(input_vector, tag_dict):
    """
    input_vector: np.array of shape (1, dim)
    tag_dict: dict {tag_name: tag_vector}
    
    Returns the tag name with the highest semantic similarity to the input.
    """
    tag_names = list(tag_dict.keys())
    tag_vectors = np.array(list(tag_dict.values()))
    
    similarities = cosine_similarity(input_vector.reshape(1, -1), tag_vectors)[0]
    max_idx = np.argmax(similarities)
    
    return tag_names[max_idx], similarities[max_idx]
