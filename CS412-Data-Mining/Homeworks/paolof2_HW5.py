import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_iris(filepath='~/Downloads/iris.data'):
    feat_names = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'class',
    ]

    df = pd.read_csv(filepath, header=None, names=feat_names)

    X = np.array(df.iloc[:, :4])

    class2idx = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2,
    }

    Y = np.array([class2idx[cls] for cls in df['class']])

    return X, Y, df


def load_test_example():
    X = np.array([
        [0, 1],
        [1, 0],
        [10, 10],
        [10, 11],
        [11, 10]
    ], dtype=float)

    Y = np.array([0, 1, 1, 1, 2])

    return X, Y


def centroids_init(X, k):
    """
    Select K points as initial centroids
    :param X: numpy.array, n_samples * n_features, data matrix
    :param k: int, number of lcusters
    :return:
    """
    
    n_samples, n_features = X.shape
    indices = np.random.choice(range(n_samples), k, replace = False)
    centroids = X[indices,:]
    

    return centroids


def kmeans(X, k):
    """
    Select K points as initial centroids
    :param X: numpy.array, float, n_samples * n_features, data matrix
    :param Y: numpy.array, int, n_samples, ground truth
    :param k: int, number of lcusters
    :return:
    """
    centroids = centroids_init(X, k)
    assignment = - np.ones(len(X), dtype=int)  # a placeholder
    sse_lst = []  # sse in each iteration


    while True:

        # Form K clusters by assigning each point to its closest centroid
        for i in range(len(X)):
            assignment[i] = np.argmin(np.linalg.norm(centroids - X[i,:], axis = 1))

        # Compute SSE:
        sse0 = 0
        for kc in range(k):
            sse0 += np.sum(np.linalg.norm(X[assignment == kc,:] - centroids[kc], axis = 1))

        # Re-compute the centroids (i.e., mean point) of each cluster
        for kc in range(k):
            centroids[kc] = np.mean(X[assignment == kc,:], axis=0) 

        # Compute SSE:
        sse1 = 0
        for kc in range(k):
            sse1 += np.sum(np.linalg.norm(X[assignment == kc,:] - centroids[kc], axis = 1))

        # Stop if already converge
        if sse0 == sse1:
            break
        else:
            sse_lst.append(sse1)


    return centroids, assignment, sse_lst


def visualize(X, assignment):
    plt.scatter(X[:, 2], X[:, 3], c=assignment)
    plt.xlabel('petal_length')
    plt.ylabel('petal_width')
    plt.title('Iris Clustering')
    plt.show()


def main():
    # load data
    X, Y, _ = load_iris('iris.data')

    np.random.seed(9)
    # Q4(a)
    centroids, assignment, sse_lst = kmeans(X,3)
    visualize(X, assignment)
    plt.close()

    # Q4(b)
    plt.plot(sse_lst)
    plt.xlabel("Iteration number")
    plt.ylabel("SSE")
    plt.savefig("Prob3-2.png")
    plt.close()
    print("SSE vs iteration: ", sse_lst)

    # Q4(c)
    SSE_k = []
    for k in range(1,11):
        centroids, assignment, sse_lst = kmeans(X,k)
        SSE_k.append(sse_lst[-1])

    plt.plot(range(1,11),SSE_k)
    plt.xlabel("Number of clusters k")
    plt.ylabel("SSE")
    plt.savefig("Prob3-3.png")
    plt.close()
    print("SSE vs number of clusters: ", SSE_k)


def test():
    X, Y = load_test_example()
    k = 2

    np.random.seed(0)  # you may want to try multiple seeds
    centroids, assignment, sse_lst = kmeans(X, k)
    print(centroids, assignment, sse_lst)

    # centroids should be [10.3333, 10.3333] and [0.5, 0.5]
    # assignment should be [0, 0, 1, 1, 1] or [1, 1, 0, 0, 0]
    # last item of sse_lst should be 2.3333

    print(nmi_score(assignment, Y))

    # nmi should be 0.3640



if __name__ == '__main__':
    test()  # test case
    main()
