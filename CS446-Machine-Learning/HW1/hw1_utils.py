import torch


def bayes_dataset(split, prefix="bayes"):
    '''
    Arguments:
        split (str): "train" or "test"

    Returns:
        X (N x d LongTensor): features of each object, X[i][j] = 0/1
        y (N LongTensor): label of each object, y[i] = 0/1

    '''
    return torch.load(f"{prefix}_{split}.pth")


def bayes_eval(prefix="bayes"):
    import hw1
    X, y = bayes_dataset("train", prefix=prefix)
    D = hw1.bayes_MAP(X, y)
    p = hw1.bayes_MLE(y)
    Xtest, ytest = bayes_dataset("test", prefix=prefix)
    ypred = hw1.bayes_classify(D, p, Xtest)
    return ypred, ytest


def gaussian_dataset(split, prefix="gaussian"):
    '''
    Arguments:
        split (str): "train" or "test"

    Returns:
        X (N x d LongTensor): features of each object, X[i][j] = 0/1
        y (N LongTensor): label of each object, y[i] = 0/1

    '''
    return torch.load(f"{prefix}_{split}.pth")


def gaussian_eval(prefix="gaussian"):
    import hw1
    X, y = gaussian_dataset("train", prefix=prefix)
    mu, sigma2 = hw1.gaussian_MAP(X, y)
    p = hw1.gaussian_MLE(y)
    Xtest, ytest = gaussian_dataset("test", prefix=prefix)
    ypred = hw1.gaussian_classify(mu, sigma2, p, Xtest)
    return ypred, ytest

import hw1
X, y = bayes_dataset("train")
D = hw1.bayes_MAP(X, y)
print(torch.sort(D[1]))

