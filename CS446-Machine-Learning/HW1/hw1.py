import torch


# import hw1_utils as utils


# Problem Naive Bayes
def bayes_MAP(X, y):
    '''
    Arguments:
        X (N x d LongTensor): features of each object, X[i][j] = 0/1
        y (N LongTensor): label of each object, y[i] = 0/1

    Returns:
        theta (2 x d Float Tensor): MAP estimation of P(X_j=1|Y=i)

    '''

    # Compute dimensions of X
    N = X.shape[0]
    d = X.shape[1]

    # Defining the tensor theta
    theta = torch.zeros((2, d))

    # Count occurrences of Y = 0 and Y = 1
    N0 = torch.sum(y == 0)
    N1 = N - N0


    theta[0] = (1 / N0)*torch.sum((torch.transpose(X,0,1)*(1-y))==1,axis=1)
    theta[1] = (1 / N1)*torch.sum((torch.transpose(X,0,1)*y)==1,axis=1)

    return theta


def bayes_MLE(y):
    '''
    Arguments:
        y (N LongTensor): label of each object

    Returns
        p (float  or scalar Float Tensor): MLE of P(Y=0)

    '''

    # Count occurrences of Y = 0 in y
    N0 = torch.sum(y == 0)

    # Return probability
    return (N0 / y.shape[0])


def bayes_classify(theta, p, X):
    '''
    Arguments:
        theta (2 x d Float Tensor): returned value of `bayes_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (N x d LongTensor): features of each object for classification, X[i][j] = 0/1

    Returns:
        y (N LongTensor): label of each object for classification, y[i] = 0/1

    '''
    N = X.shape[0]
    d = X.shape[1]
    y_pred = torch.zeros(N)

    for i in range(0, N):
        probs = []
        for Y in range(0, 2):
            sum_prob = torch.zeros(1)
            for j in range(0, d):
                if X[i][j] == 0:
                    sum_prob += (torch.log(1 - theta[Y][j]))
                else:
                    sum_prob += (torch.log(theta[Y][j]))

            probs.append(sum_prob + (1 - Y) * torch.log(p) + Y * torch.log(1 - p))
        probs = torch.stack(probs)
        y_pred[i] = torch.argmax(probs)

    return y_pred


# Problem Gaussian Naive Bayes
def gaussian_MAP(X, y):
    '''
    Arguments:
        X (N x d FloatTensor): features of each object
        y (N LongTensor): label of each object, y[i] = 0/1

    Returns:
        mu (2 x d Float Tensor): MAP estimation of mu in N(mu, sigma2)
        sigma2 (2 x d Float Tensor): MAP estimation of mu in N(mu, sigma2)'''


    # Compute dimensions of X
    N = X.shape[0]
    d = X.shape[1]

    # Defining the output tensors
    mu = torch.zeros((2, d))
    sigma2 = torch.zeros((2, d))

    y1 = []
    y0 = []
    # Go through every i-th training example
    for i in range(0, N):
        # Check if i-th value of y is 1
        if y[i] == 1:
            y1.append(X[i])
        # Similarly if y[i] = 0
        else:
            y0.append(X[i])

    y1 = torch.stack(y1)
    y0 = torch.stack(y0)

    mu[0] = torch.mean(y0, axis=0)
    mu[1] = torch.mean(y1, axis=0)
    sigma2[0] = torch.var(y0, axis=0, unbiased=False)
    sigma2[1] = torch.var(y1, axis=0, unbiased=False)

    # Return theta as a PyTorch tensor

    return mu, sigma2


def gaussian_MLE(y):
    '''
    Arguments:
        y (N LongTensor): label of each object

    Returns:
        p (float or scalar Float Tensor): MLE of P(Y=0)'''


    # Count occurrences of Y = 0 in y
    N0 = torch.sum(y == 0)

    # Return probability
    return (N0 / y.shape[0])



#def gauss(mu, sigma2, x):
    #return (1 / (torch.sqrt(2 * torch.pi * sigma2))) * (torch.e ** (-0.5 * (((x - mu)) / torch.sqrt(sigma2)) ** 2))


def gaussian_classify(mu, sigma2, p, X):
    '''
    Arguments:
        mu (2 x d Float Tensor): returned value #1 of `gaussian_MAP`
        sigma2 (2 x d Float Tensor): returned value #2 of `gaussian_MAP`
        p (float or scalar Float Tensor): returned value of `bayes_MLE`
        X (N x d FloatTensor): features of each object

    Returns:
        y (N LongTensor): label of each object for classification, y[i] = 0/1'''


    N = X.shape[0]
    y_pred = torch.zeros(N)
    probs = torch.zeros(2)

    for i in range(0, N):
        for Y in range(0, 2):
            sum_prob = torch.sum(torch.log(1 / (torch.sqrt(2 * torch.pi * sigma2[Y])))+(-0.5 * (((X[i] - mu[Y])) / torch.sqrt(sigma2[Y])) ** 2))
            probs[Y] = (sum_prob + (1 - Y) * torch.log(p) + Y * torch.log(1 - p))

        y_pred[i] = torch.argmax(probs)

    return y_pred
