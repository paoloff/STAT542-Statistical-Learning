import hw3_utils as utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (N, d).
        y_train: 1d tensor with shape (N,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (N,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().

    '''
    # Shape:
    N, d = x_train.shape

    # Initialize:
    alpha = torch.zeros(N,1, requires_grad=True)

    # Compute kernel matrix:
    kij = torch.tensor([[float(kernel(x,u)) for x in x_train] for u in x_train])


    if c == None:

        for i in range(num_iters):
            # Compute loss
            L = torch.sum(alpha) - 0.5*torch.sum(torch.outer(alpha.flatten()*y_train.flatten(), alpha.flatten()*y_train.flatten())*kij)

            # Compute gradients
            L.backward()

            # Learning rule
            alpha = alpha + lr*alpha.grad
            alpha = alpha.detach()
            alpha.clamp_(0)
            alpha.requires_grad = True


        return alpha.detach()



    else:

        for i in range(num_iters):
            # Compute loss
            L = torch.sum(alpha) - 0.5*torch.sum(torch.outer(alpha.flatten()*y_train.flatten(), alpha.flatten()*y_train.flatten())*kij)

            # Compute gradients
            L.backward()

            # Learning rule
            alpha = alpha + lr*alpha.grad
            alpha = alpha.detach()
            alpha.clamp_(0,c)
            alpha.requires_grad = True



        return alpha.detach()

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (N,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (N, d), denoting the training set.
        y_train: 1d tensor with shape (N,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (M, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (M,), the outputs of SVM on the test set.
    '''
    w = torch.sum(alpha*y_train*x_train, axis=0)
    indices = torch.nonzero(alpha)[:,0]
    nmin = torch.argmin(alpha[alpha>0])
    index_min = int(indices[nmin])
    b = 1 - y_train[index_min]*(torch.dot(w, x_train[index_min,:]))
    test = torch.sum(w*x_test, axis=1) + b

    return test

class DigitsConvNet(nn.Module):
    def __init__(self):
        '''
        Initializes the layers of your neural network by calling the superclass
        constructor and setting up the layers.

        You should use nn.Conv2d, nn.MaxPool2D, and nn.Linear
        The layers of your neural network (in order) should be
        - A 2D convolutional layer (torch.nn.Conv2d) with 7 output channels, with kernel size 3
        - A 2D maximimum pooling layer (torch.nn.MaxPool2d), with kernel size 2
        - A 2D convolutional layer (torch.nn.Conv2d) with 3 output channels, with kernel size 2
        - A fully connected (torch.nn.Linear) layer with 10 output features

        '''
        super(DigitsConvNet, self).__init__()
        torch.manual_seed(0) # Do not modify the random seed for plotting!

        # Please ONLY define the sub-modules here
        self.conv1 = nn.Conv2d(1, 7, 3)
        self.batch1 = nn.BatchNorm2d(7)
        self.maxpool = nn.MaxPool2d(2)
        self.batch2 = nn.BatchNorm2d(7)
        self.conv2 = nn.Conv2d(7, 3, 2)
        self.batch3 = nn.BatchNorm2d(3)
        self.linear = nn.Linear(3 * 2 * 2, 10)

    def forward(self, xb):
        '''
        A forward pass of your neural network.

        Note that the nonlinearity between each layer should be F.relu.  You
        may need to use a tensor's view() method to reshape outputs

        Arguments:
            self: This object.
            xb: An (N,8,8) torch tensor.

        Returns:
            An (N, 10) torch tensor
        '''
        X = torch.unsqueeze(xb, 1)
        X = self.conv1(X)
        X = self.batch1(X)
        X = F.relu(X)
        X = self.maxpool(X)
        X = self.batch2(X)
        X = F.relu(X)
        X = self.conv2(X)
        X = self.batch3(X)
        X = F.relu(X)
        batch_size, C, H, W = X.shape
        X = torch.reshape(X, (batch_size, C * H * W))
        X = self.linear(X)

        return X

def fit_and_evaluate(net, optimizer, loss_func, train, test, n_epochs, batch_size=1):
    '''
    Fits the neural network using the given optimizer, loss function, training set
    Arguments:
        net: the neural network
        optimizer: a optim.Optimizer used for some variant of stochastic gradient descent
        train: a torch.utils.data.Dataset
        test: a torch.utils.data.Dataset
        n_epochs: the number of epochs over which to do gradient descent
        batch_size: the number of samples to use in each batch of gradient descent

    Returns:
        train_epoch_loss, test_epoch_loss: two arrays of length n_epochs+1,
        containing the mean loss at the beginning of training and after each epoch
    '''
    train_dl = torch.utils.data.DataLoader(train, batch_size)
    test_dl = torch.utils.data.DataLoader(test)

    train_losses = []
    test_losses = []

    # Compute the loss on the training and validation sets at the start,
    # being sure not to store gradient information (e.g. with torch.no_grad():)

    with torch.no_grad():
        train_losses.append(utils.epoch_loss(net, loss_func, train_dl))
        test_losses.append(utils.epoch_loss(net, loss_func, test_dl))

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss

    for epoch in range(n_epochs):
        xb, yb = next(iter(train_dl))
        utils.train_batch(net, loss_func, xb, yb, optimizer)
        with torch.no_grad():
            train_losses.append(utils.epoch_loss(net, loss_func, train_dl))
            test_losses.append(utils.epoch_loss(net, loss_func, test_dl))

    return train_losses, test_losses
