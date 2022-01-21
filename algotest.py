from spca import Spca
from elasticnet import ElasticNet
import pandas as pd
import numpy as np

def main():
    # Import pitprops data
    pitprops = pd.read_csv("data/pitprops.csv", index_col=0) # EIGENVALUE DECOMPOSITION
    eigenvalues = np.linalg.eig(pitprops)
    #print(eigenvalues) # (13,13) tuple
    firstloadings = eigenvalues[0]
    #print(firstloadings)
    
    #normalized_result = Spca.SPCAalgo(firstloadings)
    #print(normalized_result)

    difference = 0.001
    threshold = 0.1
    i = 0
    k = 6
    maxit = 10000

    A =  np.zeros((maxit, len(firstloadings)))
    A[0] = firstloadings

    while difference < threshold:
        B = np.zeros((maxit, len(firstloadings))) # initialise B (10000,13)
        
        for j in range(0,k):
            B[i][j] = ElasticNet.enetoptimizer(self, 1, A[i], maxit, 0.5)( A[j] )
            #print(B[i][j])

        # Which version of equation 3 to use?
        SVD = np.linalg.svd((X.T @ X) @ B) # How to define X? Grab from elasticnet? Is part of the elasticnet equation, but also X^t@X is the covariance matrix?
        SVD = np.linalg.svd(U @ D @ np.transpose(V)) # = np.transpose(X) @ X @ B
        
        A[i+1] = U @ np.transpose(V)
        difference = A[i+1] - A[i] # Difference of two vectors? Or sum of differences over each corresponding element?
        i += 1
        #print(B[i][j])

    #normalized_loadings = B / np.linalg.norm(B)
    #print(normalized_loadings)


if __name__ == '__main__':
    main()
