import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        a = np.array(x, dtype=np.float64)  #we added explicit conversion so that we can do the calcul witht facing any problem , conversion has been done in this way : 
        # dtype : content type is float 64 (u can read about them here https://stackoverflow.com/questions/43440821/the-real-difference-between-float32-and-float64)
        n = len(weights)
        for i in range(n):
            w = np.array(weights[i], dtype=np.float64)
            b = np.array(biases[i], dtype=np.float64)
            z = a @ w + b
            if i < n - 1:
                a = np.maximum(0, z)
            else:
                a = z
        return np.round(a, 5)