import numpy as np
import silence

def sim(a: np.ndarray, b: np.ndarray) -> float:
    
    'Cosine similarity'

    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))




