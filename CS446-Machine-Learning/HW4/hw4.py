import torch
from  hw4_utils import *

def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [N, 2].
        init_c: initial centroids, shape [2, 2]. Each row is a centroid.
    
    Return:
        c: shape [2, 2]. Each row is a centroid.


    """
    if X is None:
        X, init_c = load_data()

    N = X.shape[0]
    K = 2
    Kclass = torch.zeros(N)
    c = init_c

    for n in range(n_iters):

        r = torch.zeros(N,K)
        dist = torch.zeros(N,K)

        for i in range(N):
            for k in range(K):
                dist[i,k] = torch.norm(X[i]-c[k])
            Kclass[i] = torch.argmin(dist[i])
            r[i,int(Kclass[i])] = 1

        print(r*torch.reshape(X[:,0],(N,1)))

        c[:,0] = torch.sum(r*torch.reshape(X[:,0],(N,1)), axis = 0)/torch.sum(r, axis = 0)
        c[:,1] = torch.sum(r*torch.reshape(X[:,1],(N,1)), axis = 0)/torch.sum(r, axis = 0)


    return c
    
    
