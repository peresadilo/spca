import numpy as np
import pandas as pd

def nonzeroloadings(B):
    """
    This function calculates the number of nonzero loadings. Requires the loadings matrix B, and
    returns the number of nonzero loadings per column.
    """
    nonzero_count = (B != 0).astype(int).sum(axis=0)
    return nonzero_count

def variance(X, V):
    """
    This function returns the variance (not the adjusted variance) based on the formulas provided
    by Zou, Hastie, and Tibshirani (2006). The function requires the matrices X and V as input,
    and gives the variance as output.
    """
    k = V.shape[0]
    X = X.iloc[:, :k]
    sigma = X.T @ X
    Z = V.T @ sigma @ V
    variance = np.trace(Z.T @ Z)
    return variance
    
def tex_output(tables):
    """
    This function can be used to generate LaTeX output from multiple tables. Requires one or more
    tables as input, and yields the LaTeX source code as output.
    """
    width = tables[0].shape[0]
    for table in tables:
        if(table.shape[0] != width):
            return "The provided tables do not have the same dimensions."
        else:
            pass
    
