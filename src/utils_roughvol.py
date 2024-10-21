import warnings
warnings.filterwarnings('ignore')

import numpy as np
from fbm import FBM

def fOU_generator(a: float, n: float = 0.3, h: float = 0.2, length: int = 300) -> np.ndarray:
    # X(t+1) = X(t) - a(X(t)-m) + n(W^H(t+1)-W^H(t))
    fbm_increments = np.diff(FBM(length, h).fbm())
    x0 = np.random.normal(1, 0.1)
    x0 = 0.5
    m = x0
    X = [x0]
    for i in range(length):
        p = X[i] - a*(X[i]-m) + n*fbm_increments[i]
        X.append(p)
    return np.array(X)


def price_generator(vol: np.ndarray) -> np.ndarray:
    # P(t+1) - P(t) = P(t) vol(t) (W(t+1) - W(t)) 
    length = len(vol)
    deltaW = np.random.normal(0, np.sqrt(1/(length-1)), size=length-1)
    P = np.zeros(length)
    P[0] = 1.
    for i in range(0, length-1):
        P[i+1] = P[i]*(1 + vol[i]*deltaW[i])
    return P