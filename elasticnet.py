# This class is used to define and solve the elastic net problem from the general SPCA algorithm in 
# Zou, Hastie and Tibshirani (2006).

import numpy as np
import pandas as pd
from scipy.optimize import minimize

class ElasticNet:
    
    def __init__(self):
        pass

    def enetobjective(self, B, alpha, X, l1, j):
        obj = pd.transpose(alpha[j]-B)*pd.transpose(X)*X*(alpha[j]-B)+l1*np.linalg.norm(B)^2+l1[j]*np.linalg.norm(B)
        return obj

    def enetoptimizer(self, alpha, X, maxit, l1):
        j = 0
        k = len(alpha)
        B = np.zeros(k, dtype=int)
        while(j < maxit):
            Bnew = minimize(self.enetobjective(B, alpha, X, l1, j), B, method="BFGS")
            j = j+1


