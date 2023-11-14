import hw3_utils as utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


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
        self.conv2 = nn.Conv2d(7,3,2)
        self.batch3 = nn.BatchNorm2d(3)
        self.linear = nn.Linear(3*2*2, 10)


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
        X = torch.reshape(X, (batch_size, C*H*W))
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
    train_losses.append(utils.epoch_loss(net, loss_func, train_dl))
    test_losses.append(utils.epoch_loss(net, loss_func, test_dl))

    # Train the network for n_epochs, storing the training and validation losses
    # after every epoch. Remember not to store gradient information while calling
    # epoch_loss

    for epoch in range(n_epochs):
        xb, yb = next(iter(train_dl))
        utils.train_batch(net, loss_func, xb, yb, optimizer)
        train_losses.append(utils.epoch_loss(net, loss_func, train_dl))
        test_losses.append(utils.epoch_loss(net, loss_func, test_dl))


    return train_losses, test_losses


