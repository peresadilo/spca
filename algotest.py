from spca import Spca
# from elasticnet import ElasticNet
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

def main():
    pitprops = pd.read_csv("data/pitprops.csv", index_col=0) 
    eigenvalues, eigenvectors = np.linalg.eig(pitprops)
    eigenvalues = eigenvalues[0:6]
    eigenvectors = eigenvectors[0:6, 0:6]

    Csqrt = (eigenvectors * eigenvalues**0.5) @ eigenvectors.T
    
    difference = 10 # Intial diff has to > threshold
    threshold = 3.8 # Random default value
    i = 0 # Initialize i
    k = 6 # Default value from paper
    maxit = 600 # Random default value
    lambda1 = np.array([0.06, 0.16, 0.1, 0.5, 0.5, 0.5]) # Default value from paper

    A =  np.zeros((maxit, k))
    Atemp = eigenvectors

    B = np.zeros((606, 6))
    B_temp = np.zeros((6,k))

    while difference > threshold and i < maxit:
        for j in range(0,k):
            elastic_net_solver = ElasticNet(alpha=1, l1_ratio=lambda1[j], fit_intercept=False, max_iter=1e6).fit(Csqrt, Csqrt @ Atemp[:, 1])
            B_temp[j] = np.array(elastic_net_solver.coef_)
        B[i:(i+k)] = B_temp
        U, D, vh = np.linalg.svd(pitprops.iloc[0:6,0:6]*pitprops.iloc[0:6,0:6]*B[i], full_matrices=True)
        difference = np.linalg.norm((U @ np.transpose(vh))-Atemp)
        Atemp = (U @ np.transpose(vh))
        i += 1
            
    print("Optimization terminated succesfully \n")

    print("Normalized Loadings")
    normalized_loadings = B[i:(i+6)] / np.linalg.norm(B[i:(i+6)])
    print(normalized_loadings)

    # Open questions:
    # Eerste 6 loadings? 6x13 of 6x1?
    # Hoe SVD van geniedata in equation 3 berekenen?
    # Which version of equation 3 to use?
    # SVD = np.linalg.svd((X.T @ X) @ B) # How to define X? Grab from elasticnet? Is part of the elasticnet equation, but also X^t@X is the covariance matrix?
    # SVD = np.linalg.svd(U @ D @ np.transpose(V)) # = np.transpose(X) @ X @ B
    # Correlation matrix * vector 
    # (20*13)(13*1) # First 6 loadings or all 13 loadings, how to subset the 

if __name__ == '__main__':
    main()