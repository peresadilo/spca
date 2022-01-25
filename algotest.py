from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

# To-do after results are correct
# - Fix code layout (i.e. transfer algotest.py to notebook and class)
# - PEP8
# - Implement gene dataset
# - Implement self-built elasticnet optimizer 

def main():
    pitprops = pd.read_csv("data/pitprops.csv", index_col=0) 
    eigenvalues, eigenvectors = np.linalg.eig(pitprops)

    eigenvalues = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors.argsort()[::-1]

    eigenvalues = eigenvalues
    eigenvectors = eigenvectors

    Csqrt = (eigenvectors * eigenvalues**0.5) @ eigenvectors.T

    difference = 1 # Intial diff has to > threshold
    difference_b = 1
    threshold = 1e-6 # Random default value
    i = 0 # Initialize i
    k = 6 # Default value from paper
    maxit = 1000 # Random default value
    lambda1 = np.array([0.06, 0.16, 0.1, 0.5, 0.5, 0.5]) # Default value from paper
    lambda1 = lambda1

    Atemp = eigenvectors[:, :k]

    B = np.zeros((10600, 13))
    B_temp = np.zeros((k,13))

    #while difference > threshold and difference_b > threshold and i < maxit:
    while i < maxit:
        for j in range(0,k):
            elastic_net_solver = ElasticNet(alpha=lambda1[j], l1_ratio=0.5, fit_intercept=False, max_iter=1e6).fit(Csqrt, Csqrt @ Atemp[:, j])
            B_temp[j] = np.array(elastic_net_solver.coef_)
        B[i:(i+k)] = B_temp

        U, D, vh = np.linalg.svd((pitprops.T @ pitprops) @ np.transpose(B[i:(i+k)]))
        U =  U[:, :k]
        vh = vh[:k]
        difference = np.linalg.norm((U @ np.transpose(vh))-Atemp)
        difference_b = 1 if i < 10 else np.linalg.norm((B[i:(i+k)])-(B[(i-6):(i+k-6)]))
        Atemp = (U @ np.transpose(vh))
        print(difference_b)
        i += 1
            
    print("Optimization terminated succesfully \n")

    print("Normalized Loadings")
    normalized_loadings = B[i:(i+6)] / np.linalg.norm(B[i:(i+6)])
    print(normalized_loadings)

if __name__ == '__main__':
    main()