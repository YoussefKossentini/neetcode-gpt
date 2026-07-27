import numpy as np
from numpy.typing import NDArray


class Solution:
 def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    eps = 1e-5

    moy = np.mean(x)
    sigma_sqrd = np.mean((x - moy) ** 2)
    x_hat = (x - moy) / np.sqrt(sigma_sqrd + eps)
    out = gamma * x_hat + beta
    return np.round(out, 5)