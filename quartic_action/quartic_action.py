import numpy as np
from numpy.linalg import det, inv
from typing import Dict, Callable, List
from itertools import product
from collections import Counter
import math


def create_random_quartic_coupling(n: int, rng: Callable[[], float]) -> np.ndarray:
    result = np.zeros(shape=(n, n, n, n))

    cache: Dict[FrozenCounter[List[int]], float] = dict()
    indices = list(range(n))
    for (a, b, c, d) in product(indices, indices, indices, indices):
        counter = FrozenCounter([a, b, c, d])
        if counter not in cache.keys():
            cache[counter] = rng()
        result[a, b, c, d] = cache[counter]

    return result


class FrozenCounter(Counter):

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class QuarticProbabilityDistribution:

    def __init__(self, K: np.ndarray, V: np.ndarray, epsilon: float):
        self.K = K
        self.K_inv = inv(K)
        self.V = V
        self.epsilon = epsilon
        self.N = len(K[0, :])

        acc = 0
        indices = list(range(self.N))
        for (a, b, c, d) in product(indices, indices, indices, indices):
            acc += V[a, b, c, d] * self.K_inv[a, b] * self.K_inv[c, d]
        self.normalization_constant = math.sqrt(det(2 * math.pi * K)) * (1 - (1/8) * epsilon * acc)

    def compute(self, z: np.ndarray):
        assert(len(z) == len(self.K[0, :]))

        indices = list(range(self.N))
        acc1 = 0
        for (a, b) in product(indices, indices):
            acc1 += self.K_inv[a, b] * z[a] * z[b]
        acc2 = 0
        for (a, b, c, d) in product(indices, indices, indices, indices):
            acc2 += self.V[a, b, c, d] * z[a] * z[b] * z[c] * z[d]
        action = (1/2) * acc1 + (self.epsilon / 24) * acc2

        result = math.exp(-action) / self.normalization_constant
        return result
