import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
                            
        x=np.array(x) #we have to transtype the list onto a vector 
        w1=np.array(W1)
        b1=np.array(b1)
        w2=np.array(W2)
        b2=np.array(b2)
        y_true=np.array(y_true)

        z1=x @ w1.T +b1 #mat.T = the transpose of the matrice "mat"
        a1=np.maximum(0 , z1)
        z2=a1 @ w2.T +b2
        pred=z2
        loss=np.mean((z2-y_true)**2)

        if y_true.ndim>0:
            n=len(y_true)
        else :
            n=1

        dpred = 2 * (pred - y_true) / n
        dW2 = np.outer(dpred, a1)
        db2 = dpred

        da1 = w2.T @ dpred
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1, x)
        db1 = dz1

        return {'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }
        


        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

