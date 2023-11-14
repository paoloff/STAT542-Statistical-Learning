import torch
import hw2_utils as utils
import matplotlib.pyplot as plt


'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

'''

# Problem Linear Regression
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    n, d = X.shape
    Xe = torch.zeros(n, d+1)
    Xe[:, 1:d+1] = X
    Xe[:, 0] = torch.ones(n)
    w = torch.zeros(d+1, 1)
    for i in range(num_iter):
        dR = (2/n)*torch.sum((torch.matmul(Xe, w)-Y)*Xe, axis=0)
        w = w - lrate*torch.reshape(dR, (d+1,1))

    return w


def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    n, d = X.shape
    Xe = torch.zeros(n, d + 1)
    Xe[:, 1:d+1] = X
    Xe[:, 0] = torch.ones(n)
    return torch.matmul(torch.pinverse(Xe), Y)


def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()
    d = X.shape[1]
    w = linear_normal(X, Y)
    pred = torch.matmul(X, w[1:d+1]) + w[0]
    plt.plot(X, Y, label='data')
    plt.plot(X, pred, label='prediction')
    return plt.gcf()



# Problem Logistic Regression
def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float, default: 0.01): learning rate
        num_iter (int, default: 1000): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w'
    
    '''
    n, d = X.shape
    Xe = torch.zeros(n, d+1)
    Xe[:, 1:d+1] = X
    Xe[:, 0] = torch.ones(n)
    w = torch.zeros(d+1, 1)
    for i in range(num_iter):
        dR = (1/n)*torch.sum((1/(1+torch.exp(-Y*torch.matmul(Xe, w))))*torch.exp(-Y*torch.matmul(Xe, w))*((-Y)*Xe), axis=0)
        w = w - lrate*torch.reshape(dR, (d+1,1))

    return w


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_logistic_data()
    wlog = logistic(X, Y)
    x = torch.linspace(torch.min(X[:,0]),torch.max(X[:,1]), 20)
    wlin = linear_gd(X, Y)
    Ydb_log = -(wlog[0]+wlog[1]*x)/wlog[2]
    Ydb_lin = -(wlin[0] + wlin[1] * x) / wlin[2]
    color = ['red' if y == 1 else 'blue' for y in Y]
    plt.scatter(X[:,0], X[:,1], color=color)
    plt.plot(x, Ydb_log, label='logistic DB')
    plt.plot(x, Ydb_lin, label='LS DB')
    plt.legend()


    return plt.gcf()
