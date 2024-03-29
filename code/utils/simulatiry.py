import numpy as np

class CosinSimilarity:
    def __call__(self, v1, v2):
        return np.dot(v1, v2)
    