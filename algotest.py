from spca import Spca
from elasticnet import ElasticNet
import pandas as pd
import numpy as np

def main():
    # Import pitprops data
    pitprops = pd.read_csv("data/pitprops.csv", index_col=0) 
    eigenvalues = np.linalg.eig(pitprops)
    firstloadings = eigenvalues[0]
    
    #normalized_result = Spca.SPCAalgo(firstloadings)
    #print(normalized_result)

    difference = 10 # Intial diff has to > threshold
    threshold = 0.1 # Random default value
    i = 0 # Initialize i
    k = 6 # Default value from paper
    maxit = 10000 # Random default value
    lambda1 = np.array[0.06, 0.16, 0.1, 0.5, 0.5, 0.5] # Default value from paper
   
    A =  np.zeros((maxit, len(firstloadings)))
    A[0] = firstloadings

    while difference < threshold:
        B = np.zeros((maxit, len(firstloadings))) # initialise B (10000,13)
        
        for j in range(0,k):
            #B[i][j] = ElasticNet.enetoptimizer(self, 1, A[i], maxit, lambda1)( A[j] )
            
            #print(B[i][j])

        U, D, vh = np.linalg.svd(pitprops.iloc[0:6,0:6]*B[i], full_matrices=True)
        A[i+1] = U @ np.transpose(vh)
        difference = np.linalg.norm(A[i+1]-A[i]) # Difference of two vectors? Or sum of differences over each corresponding element?
        i += 1
        #print(B[i][j])

    normalized_loadings = B / np.linalg.norm(B)
    print(normalized_loadings)

    # Open questions:
    # Eerste 6 loadings? 6x13 of 6x1?
    # Hoe SVD van geniedata in equation 3 berekenen?
    # Which version of equation 3 to use?
    # SVD = np.linalg.svd((X.T @ X) @ B) # How to define X? Grab from elasticnet? Is part of the elasticnet equation, but also X^t@X is the covariance matrix?
    # SVD = np.linalg.svd(U @ D @ np.transpose(V)) # = np.transpose(X) @ X @ B
    # Correlation matrix * vector 
    # (20*13)(13*1) # First 6 loadings or all 13 loadings, how to subset the 

if _name_ == '_main_':
    main()