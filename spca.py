import numpy as np
import pandas as pd

# This class will be used for all main computations w.r.t. SPCA; Following Zou, Hastie and Tibshirani (2006), we can divide the SPCA algorithm in 5 different steps;
# (1) Define A using V; here, V are the loadings of the first k PCs. Depending on the dataset we can calculate V by using the single value decomposition 
# (when we have the full dataset) or by using the eigen decomposition (in case we only have the correlation matrix). Both the SVD and ED are available in Numpy
# (2) Solve the elastic net problem given in Zou, Hastie and Tibshirani (2006); we can choose here to either program an elasticnet optimizer ourselves (should be
# doable and a nice challenge) or use the built-in ElasticNet function from scikit-learn
# (3) Compute B using the SVD from Zou, Hastie and Tibshirani (2006), then update A using A=UV^T
# (4) Repeat 2-3 until convergence
# (5) Normalize V
# After those 5 steps, we also want to compute some measures like number of nonzero loadings, variance and cumulative variance. We also need to think how we want
# to visualize the SPCA outcomes for the gene dataset, since we can't simply show ~16,000 loadings

class Spca:
    
    def __init__(self) -> None:
        pass

    def SPCAalgo(V):
        A[0] = V
        difference = 10
        threshold = 0.01
        i = 0

        while difference < threshold:
            B = np.zeros(k) # initialise B
            for j in range(1,k):
                B[j] = elasticnet.enetoptimizer(self, alpha, X, maxit, l1)( A[j] )
            SVD = U @ D @ np.transpose(V) # = np.transpose(X) @ X @ B
            A[i+1] = U @ np.transpose(V)
            difference = A[i+1] - A[i]
            i += 1

        normalized_loadings = B / np.linalg.norm(B)

        return normalized_loadings



