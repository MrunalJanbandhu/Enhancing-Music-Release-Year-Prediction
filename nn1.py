import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
SCALE_FAC = 100
# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x, derivative=False):
    if derivative:
        x = np.where(x < 0, 0, 1)
        return x
    else:
        return np.maximum(0, x)


class Net(object):
    '''
    '''

    def __init__(self, num_layers, num_units):
        """
        Initialize the neural network.
        Create weights and biases.
        Here, we have provided an example structure for the weights and biases.
        It is a list of weight and bias matrices, in which, the
        dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
        weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
        biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]
        Please note that this is just an example.
        You are free to modify or entirely ignore this initialization as per your need.
        Also you can add more state-tracking variables that might be useful to compute
        the gradients efficiently.
        Parameters
        ----------
            num_layers : Number of HIDDEN layers.
            num_units : Number of units in each Hidden layer.
        """
        self.num_layers = num_layers
        self.num_units = num_units
        self.biases = []
        self.weights = []
        for i in range(num_layers):

            if i == 0:
                # Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
            else:
                # Hidden layers
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

    def __call__(self, X):
        """
        Forward propagate the input X through the network,
        and return the output.
        Note that for a classification task, the output layer should
        be a softmax layer. So perform the computations accordingly
        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
        Returns
        ----------
            y : Output of the network, numpy array of shape m x 1
        """
        a = X
        weights = self.weights
        biases = self.biases
        activations = [a]
        pre_activations = [a]
        for i, (w, b) in enumerate(zip(weights, biases)):
            h = np.dot(a, w) + b.T
            if i < len(weights) - 1:
                a = relu(h, False)
            else:
                a = h
            activations.append(a)
            h = ((h - np.mean(h)) / np.std(h))
            pre_activations.append(h)  # batch normalization(for current batch)

        pre_activations = pre_activations
        self.activations = activations
        self.pre_activations = pre_activations
        self.pred = a

        return a

    def backward(self, X, y, lamda):
        """
        Compute and return gradients loss with respect to weights and biases.
        (dL/dW and dL/db)
        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
            y : Output of the network, numpy array of shape m x 1
            lamda : Regularization parameter.
        Returns
        ----------
            del_W : derivative of loss w.r.t. all weight values (a list of matrices).
            del_b : derivative of loss w.r.t. all bias values (a list of vectors).
        Hint: You need to do a forward pass before performing backward pass.
        """
        pred = self.__call__(X)

        weights = self.weights
        biases = self.biases
        Z = self.pre_activations
        A = self.activations

        batch_sz = len(X)
        H = self.num_layers
        db = [np.zeros(b.shape) for b in biases]
        dW = [np.zeros(w.shape) for w in weights]
        for L in range(H, -1, -1):
            if L != H:
                delta = relu(Z[L + 1], True) * np.dot(delta, weights[L + 1].T)
            else:
                delta = (A[L + 1] - y) / batch_sz
                lamda
            db[L] = np.sum(delta, axis=0)[:, None] + (biases[L] * lamda) / batch_sz
            if L != 0:
                dW[L] = (np.dot(A[L].T, delta) + (weights[L] * lamda)) / batch_sz
            else:
                dW[L] = (np.dot(X.T, delta) + (weights[L] * lamda)) / batch_sz
        return dW, db


class Optimizer(object):
    '''
    '''

    def __init__(self, learning_rate):

        self.learning_rate = learning_rate

    def step(self, weights, biases, delta_weights, delta_biases):
        '''
        Parameters
        ----------
            weights: Current weights of the network.
            biases: Current biases of the network.
            delta_weights: Gradients of weights with respect to loss.
            delta_biases: Gradients of biases with respect to loss.

        '''
        weights_updated = []
        biases_updated = []
        for w, w1 in zip(weights, delta_weights):
            weights_updated.append(w - (self.learning_rate * w1))
        for b, b1 in zip(biases, delta_biases):
            biases_updated.append(b - (self.learning_rate * b1))

        return weights_updated, biases_updated


def loss_mse(y, y_hat):
    '''
    Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.
    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1
    Returns
    ----------
        MSE loss between y and y_hat.
    '''
    # print(np.shape(y))
    # print(np.shape(y_hat[0]))
    # return (np.sum((y-y_hat)**2))/np.shape(y)[0]
    return np.mean(np.square(y - y_hat))


def loss_regularization(weights, biases):
    '''
    Compute l2 regularization loss.
    Parameters
    ----------
        weights and biases of the network.
    Returns
    ----------
        l2 regularization loss 
    '''
    ans = 0
    for w in weights:
        ans = ans + np.sum(np.square(w))
    for b in biases:
        ans = ans + np.sum(np.square(b))
    return ans


def loss_fn(y, y_hat, weights, biases, lamda):
    '''
    Compute loss =  loss_mse(..) + lamda * loss_regularization(..)
    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1
        weights and biases of the network
        lamda: Regularization parameter
    Returns
    ----------
        l2 regularization loss 
    '''
    return loss_mse(y, y_hat) + (lamda / 2 * loss_regularization(weights, biases))


def rmse(y, y_hat):
    '''
    Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.
    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1
    Returns
    ----------
        RMSE between y and y_hat.
    '''
    return np.sqrt(loss_mse(y, y_hat))


def cross_entropy_loss(y, y_hat):
    '''
    Compute cross entropy loss
    Parameters
    ----------
        y : targets, numpy array of shape m x 1
        y_hat : predictions, numpy array of shape m x 1
    Returns
    ----------
        cross entropy loss
    '''
    raise NotImplementedError


def train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
):
    """
    In this function, you will perform following steps:
        1. Run gradient descent algorithm for `max_epochs` epochs.
        2. For each bach of the training data
            1.1 Compute gradients
            1.2 Update weights and biases using step() of optimizer.
        3. Compute RMSE on dev data after running `max_epochs` epochs.
    Here we have added the code to loop over batches and perform backward pass
    for each batch in the loop.
    For this code also, you are free to heavily modify it.
    """

    m = train_input.shape[0]
    plt_train_loss = []
    plt_dev_loss = []
    for e in range(max_epochs):
        epoch_loss = 0.
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i + batch_size]
            batch_target = train_target[i:i + batch_size]

            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, lamda)
            pred = net.pred

            # Get updated weights based on current weights and gradients
            weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
            epoch_loss += batch_loss / batch_size

        print(epoch_loss)
        train_pred=net(train_input)
        dev_pred = net(dev_input) * SCALE_FAC
        train_mse=loss_mse(train_target,train_pred)
        dev_rmse = rmse(dev_target, dev_pred)
        plt_train_loss.append(train_mse)
        plt_dev_loss.append(dev_rmse)

    plt.plot(plt_train_loss)
    plt.title(f"Training MSE Loss batch size {batch_size}")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training MSE Loss")
    
    plt.savefig(f"train_{batch_size}.png")
    plt.cla()
    plt.plot(plt_dev_loss)
    plt.title(f"Dev RMSE Loss batch size {batch_size}")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Dev RMSE Loss")
    plt.savefig(f"dev_{batch_size}.png")
        # Write any early stopping conditions required (only for Part 2)
        # Hint: You can also compute dev_rmse here and use it in the early
        # 		stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    dev_pred = net(dev_input) * SCALE_FAC
    dev_rmse = rmse(dev_target, dev_pred)
    print(dev_pred, dev_target)
    print('RMSE on dev data: {:.5f}'.format(dev_rmse))
    return dev_rmse



def get_test_data_predictions(net, inputs):
    """
    Perform forward pass on test data and get the final predictions that can
    be submitted on Kaggle.
    Write the final predictions to the part2.csv file.
    Parameters
    ----------
        net : trained neural network
        inputs : test input, numpy array of shape m x d
    Returns
    ----------
        predictions (optional): Predictions obtained from forward pass
                                on test data, numpy array of shape m x 1
    """
    pred = net.__call__(inputs) * SCALE_FAC
    with open("part2.csv", 'w') as file:
        writer = csv.writer(file, delimiter=",")
        writer.writerow(['Id', 'Predictions'])
        for idx, row in enumerate(pred):
            writer.writerow([idx + 1, row[0]])


def normalise(x):
    return (x - (np.mean(x, axis=0))) / np.std(x, axis=0)



def read_data():
    """
    Read the train, dev, and test datasets
    """
    train = pd.read_csv("regression/data/train.csv")
    dev = pd.read_csv("regression/data/dev.csv")
    test = pd.read_csv("regression/data/test.csv")

    train_input = normalise(train.values[:, 1:])
    train_target = train.values[:, 0:1] / SCALE_FAC
    dev_input = normalise(dev.values[:, 1:])
    dev_target = dev.values[:, 0:1]
    test_input = normalise(test.values)

    return train_input, train_target, dev_input, dev_target, test_input


def main():
    # Hyper-parameters
    max_epochs = 100
    batch_size = 64
    learning_rate = 0.001
    num_layers = 1
    num_units = 64
    lamda = 0.1
    train_input, train_target, dev_input, dev_target, test_input = read_data()

    # testing loss function
    net = Net(num_layers, num_units)

    optimizer = Optimizer(learning_rate)

    loss = train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
    )

    get_test_data_predictions(net, test_input)


if __name__ == '__main__':
    main()
