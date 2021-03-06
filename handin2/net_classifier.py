import numpy as np
from sklearn.utils import shuffle
from sklearn.utils import resample
from termcolor import colored


def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes 
    
    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc

def relu_derivative(x):
    return 1. * (x > 0)
    # This is the same as :
    # x[x <= 0] = 0
    # x[x > 0] = 1
    # return x


def softmax(X):
    """ 
    You can take this from handin I
    Compute the softmax of each row of an input matrix (2D numpy array). 
    
    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate 
    
    More precisely this is what you must do.
    
    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that
    
    You can do all of it without for loops using numpys vectorized operations.

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    ### YOUR CODE HERE no for loops please


    # Copied from handin 1
    maxs = np.max(X, axis=1)
    diff_mat = (X.T - maxs).T
    exp_mat = np.exp(diff_mat)
    sum_mat = np.sum(exp_mat, axis=1)
    logsum = (np.log(sum_mat).T + maxs).T
    res = np.exp((X.T - logsum).T)

    ### END CODE
    return res

def relu(x):
    """ Compute the relu activation function on every element of the input
    
        Args:
            x: np.array
        Returns:
            res: np.array same shape as x
        Beware of np.max and look at np.maximum
    """
    ### YOUR CODE HERE
    res = np.maximum(x, 0)
    ### END CODE
    return res

def make_dict(W1, b1, W2, b2):
    """ Trivial helper function """
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def get_init_params(input_dim, hidden_size, output_size):
    """ Initializer function using he et al Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

    Args:
      input_dim: int
      hidden_size: int
      output_size: int
    Returns:
       dict of randomly initialized parameter matrices.
    """
    W1 = np.random.normal(0, np.sqrt(2./(input_dim+hidden_size)), size=(input_dim, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.normal(0, np.sqrt(4./(hidden_size+output_size)), size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}



class NetClassifier():
    
    def __init__(self):
        """ Trivial Init """
        self.params = None

        self.hist = []
        self.hist = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }

        self.val_loss = []
        self.train_loss = []
        self.train_acc = []
        self.val_acc = []

    def predict(self, X, params=None):
        """ Compute class prediction for all data points in class X
        
        Args:
            X: np.array shape n, d
            params: dict of params to use (if none use stored params)
        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params
        pred = None

        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        ### YOUR CODE HERE

        z1 = X.dot(W1) + b1
        a1 = relu(z1)
        z2 = a1.dot(W2) + b2
        pred = softmax(z2)

        ### END CODE
        return pred

    def score(self, X, y, params=None):
        """ Compute accuracy of model on data X with labels y

        Args:
            X: np.array shape n, d
            y: np.array shape n, 1
            params: dict of params to use (if none use stored params)

        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params
        acc = None
        ### YOUR CODE HERE
        y_hat = np.argmax(self.predict(X), axis=1)
        acc = np.sum(y == y_hat)/len(y)
        ### END CODE
        return acc
    
    @staticmethod
    def cost_grad(X, y, params, reg=1.0):
        """ Compute cost and gradient of neural net on data X with labels y using weight decay parameter c
        You should implement a forward pass and store the intermediate results
        and the implement the backwards pass using the intermediate stored results

        Use the derivative for cost as a function for input to softmax as derived above

        Args:
            X: np.array shape n, self.input_size
            y: np.array shape n, 1
            params: dict with keys (W1, W2, b1, b2)
            reg: float - weight decay regularization weight
            params: dict of params to use for the computation

        Returns
            cost: scalar - average cross entropy cost
            dict with keys
            d_w1: np.array shape w1.shape, entry d_w1[i, j] = \partial cost/ \partial w1[i, j]
            d_w2: np.array shape w2.shape, entry d_w2[i, j] = \partial cost/ \partial w2[i, j]
            d_b1: np.array shape b1.shape, entry d_b1[1, j] = \partial cost/ \partial b1[1, j]
            d_b2: np.array shape b2.shape, entry d_b2[1, j] = \partial cost/ \partial b2[1, j]

        """

        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        labels = one_in_k_encoding(y, W2.shape[1]) # shape n x k
        # One data point: O(  # weights and biases)
        ### YOUR CODE HERE - FORWARD PASS - compute regularized cost and store relevant values for backprop
        a0 = X.dot(W1) + b1
        a1 = relu(a0)
        a2 = a1.dot(W2)+b2
        Y_hat = softmax(a2)

        ### END CODE
        
        ### YOUR CODE HERE - BACKWARDS PASS - compute derivatives of all (regularized) weights and bias, store them in d_w1, d_w2' d_w2, d_b1, d_b2
        R = reg * (np.sum(np.square(W1.copy())) + np.sum(np.square(W2.copy())))
        cost = 1 / len(X) * -np.sum(labels * np.log(Y_hat)) + R

        # backward
        delta3 = (-labels+Y_hat)/len(X)

        dW2 = np.dot(a1.T,delta3) + (reg * 2*W2)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = (delta3.dot(W2.T) * relu_derivative(a0))

        dW1 = X.T.dot(delta2) + (reg * 2*W1)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        ### END CODE
        # the return signature
        return cost, {'d_w1': dW1, 'd_w2': dW2, 'd_b1': db1, 'd_b2': db2}



    def fit(self, X_train, y_train, X_val, y_val, init_params, batch_size=30, lr=0.1, reg=1e-4, epochs=30):
        """ Run Mini-Batch Gradient Descent on data X, Y to minimize the in sample error (1/n)Cross Entropy for Neural Net classification
        Printing the performance every epoch is a good idea to see if the algorithm is working
    
        Args:
           X_train: numpy array shape (n, d) - the training data each row is a data point
           y_train: numpy array shape (n,) int - training target labels numbers in {0, 1,..., k-1}
           X_val: numpy array shape (n, d) - the validation data each row is a data point
           y_val: numpy array shape (n,) int - validation target labels numbers in {0, 1,..., k-1}
           init_params: dict - has initial setting of parameters
           lr: scalar - initial learning rate
           batch_size: scalar - size of mini-batch
           epochs: scalar - number of iterations through the data to use

        Sets: 
           params: dict with keys {W1, W2, b1, b2} parameters for neural net
           history: dict:{keys: train_loss, train_acc, val_loss, val_acc} each an np.array of size epochs of the the given cost after every epoch
        """

        n = X_train.shape[0]
        W1 = init_params['W1']
        b1 = init_params['b1']
        W2 = init_params['W2']
        b2 = init_params['b2']

        validation_gradient = 0
        cost_dictionary = None


        ### YOUR CODE HERE
        for i in range(epochs):
            X_shuff, Y_shuff = shuffle(X_train, y_train)

            epoch_loss = []
            for j in range(batch_size):
                X_mini = resample(X_shuff, n_samples=batch_size, random_state=0)
                Y_mini = resample(Y_shuff, n_samples=batch_size, random_state=0)

                cost_dictionary = self.cost_grad(X_mini, Y_mini, params={'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, reg=reg)

                W1 -= lr * cost_dictionary[1]["d_w1"]
                W2 -= lr * cost_dictionary[1]["d_w2"]
                b1 -= lr * cost_dictionary[1]["d_b1"]
                b2 -= lr * cost_dictionary[1]["d_b2"]
                epoch_loss.append(cost_dictionary[0])



            self.params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
            validation_gradient = self.cost_grad(X_val, y_val, params={'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, reg=reg)
            cost_dictionary = self.cost_grad(X_train, y_train, params={'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}, reg=reg)
            history = {
                'train_loss': cost_dictionary[0],
                'train_acc': self.score(X_train, y_train, params=self.params),
                'val_loss': validation_gradient[0],
                'val_acc': self.score(X_val, y_val, params=self.params),
            }
            print("Epoch: ", i, "->", history)
            self.hist['train_loss'].append(history['train_loss'])
            self.hist['train_acc'].append(history['train_acc'])
            self.hist['val_loss'].append(history['val_loss'])
            self.hist['val_acc'].append(history['val_acc'])


            improvmnet = np.abs(epoch_loss[len(epoch_loss)-1] - (cost_dictionary[0]))
            print("Epoch: ",i, "Improvment in loss: ", improvmnet)
            # Every 5th run excluding the first
            if i > 1 and i % 2 == 0:
                # take a look at validation set improvment in loss... len -1 is the same gradient
                epoch_loss = []
                print("Improved loss: ", improvmnet)

                # if it is not significant we are done
                if improvmnet < 0.07:
                    print("Stop in iteration:", i)
                    return self.params

        ### END CODE

    
        

def numerical_grad_check(f, x, key):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-5
    # d = x.shape[0]
    cost, grad = f(x)
    grad = grad[key]
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:    
        dim = it.multi_index    
        print(dim)
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        #print('cplus cminus', cplus, cminus, cplus-cminus)
        #print('dim, grad, num_grad, grad-num_grad', dim, grad[dim], num_grad, grad[dim]-num_grad)
        print("d:",grad[dim]/num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()

def test_grad():
    stars = '*'*5
    print(stars, 'Testing  Cost and Gradient Together')
    input_dim = 7
    hidden_size = 1
    output_size = 3
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)

    nc = NetClassifier()
    X = np.random.randn(7, input_dim)
    y = np.array([0, 1, 2, 0, 1, 2, 0])

    f = lambda z: nc.cost_grad(X, y, params, reg=1.0)
    print('\n', stars, 'Test Cost and Gradient of b2', stars)
    numerical_grad_check(f, params['b2'], 'd_b2')
    print(stars, 'Test Success', stars)
    
    print('\n', stars, 'Test Cost and Gradient of b1', stars)
    numerical_grad_check(f, params['b1'], 'd_b1')
    print('Test Success')


    print('\n', stars, 'Test Cost and Gradient of w2', stars)
    numerical_grad_check(f, params['W2'], 'd_w2')
    print('Test Success')

    print('\n', stars, 'Test Cost and Gradient of w1', stars)
    numerical_grad_check(f, params['W1'], 'd_w1')
    print('Test Success')


if __name__ == '__main__':
    input_dim = 3
    hidden_size = 5
    output_size = 4
    batch_size = 7
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)
    X = np.random.randn(batch_size, input_dim)

    Y = np.array([0, 1, 2, 0, 1, 2, 0])
    nc.cost_grad(X, Y, params, reg=0)
    test_grad()
