from scipy.optimize import minimize
import numpy as np
import pandas as pd

def objective(B, l2, l1, X, Y):
    obj = np.linalg.norm(Y - X @ B)**2 + l1 * np.linalg.norm(B)+ l2 * np.linalg.norm(B)**2
    return obj

def solver(l2, l1, X, Y):
    B_start = np.zeros((1, 13))
    B_new = minimize(objective, x0=B_start, args=(l2, l1, X, Y), tol=1e6, method="BFGS")
    return B_new.x
