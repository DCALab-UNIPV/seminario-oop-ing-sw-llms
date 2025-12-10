import numpy as np

def softmax(xs):
    return np.exp(xs) / sum(np.exp(xs))
