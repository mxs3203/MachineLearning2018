import numpy as np
import os
from h1_util import numerical_grad_check
from sklearn.utils import shuffle
from sklearn.utils import resample


def softmax(X):
    """ 
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
    maxs = np.max(X, axis=1)
    diff_mat = (X.T-maxs).T
    exp_mat = np.exp(diff_mat)
    sum_mat = np.sum(exp_mat, axis = 1)
    logsum = (np.log(sum_mat).T+maxs).T
    res = np.exp((X.T-logsum).T)
    
    ### END CODE
    return res

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
    
class SoftmaxClassifier():

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.W = None
        
    def cost_grad(self, X, y, W):
        """ 
        Compute the average cross entropy cost and the gradient under the softmax model 
        using data X, Y and weight vector W.
        
        the functions np.nonzero, np.sum, np.dot (@), may come in handy
        Args:
           X: numpy array shape (n, d) float - the data each row is a data point
           y: numpy array shape (n, ) int - target values in 0,1,...,k-1
           W: numpy array shape (d x K) float - weight matrix
        Returns:
            totalcost: Average Negative Log Likelihood of w
            gradient: The gradient of the average Negative Log Likelihood at w 
        """

        Yk = one_in_k_encoding(y, self.num_classes)
        input_size = X.shape[0]
        cost = np.nan
        grad = np.zeros(W.shape) * np.nan
        soft = np.log(softmax(np.dot(X, W)))
        cost = -np.sum((Yk.T.dot(soft[Yk == 1])) / input_size)

        grad = -np.transpose(X).dot(Yk - softmax(X.dot(W))) / input_size

        ### END CODE
        return cost, grad


    def fit(self, X, Y, W=None, lr=0.01, epochs=10, batch_size=2):
        """
        Run Mini-Batch Gradient Descent on data X,Y to minimize the in sample error (1/n)NLL for softmax regression.
        Printing the performance every epoch is a good idea to see if the algorithm is working
    
        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
           W: numpy array shape (d x K)
           lr: scalar - initial learning rate
           batchsize: scalar - size of mini-batch
           epochs: scalar - number of iterations through the data to use

        Sets: 
           W: numpy array shape (d, K) learned weight vector matrix  W
           history: list/np.array len epochs - value of cost function after every epoch. You know for plotting
        """
        if W is None: W = np.zeros((X.shape[1], self.num_classes))
        history = []
        ### YOUR CODE HERE
        n = X.shape[0]
        for j in range(epochs):
            X_shuff, Y_shuff = shuffle(X, Y)
            for i in range(n // batch_size):
                X_mini = resample(X_shuff, n_samples=batch_size, random_state=0)
                Y_mini = resample(Y_shuff, n_samples=batch_size, random_state=0)
                cost, grad = self.cost_grad(X_mini, Y_mini, W)
                history.append(cost)
                W -= lr * grad
                #print("Cost", cost)
                #print("W", W)
        ### END CODE
        self.W = W
        self.history = history
        

    def score(self, X, Y):
        """ Compute accuracy of classifier on data X with labels Y

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
           Y: numpy array shape (n,) int - target labels numbers in {0, 1,..., k-1}
        Returns:
           out: float - accuracy
        """
        out = 0
        ### YOUR CODE HERE 1-4 lines
        predictions = self.predict(X)
        out = np.sum(predictions == Y) / X.shape[0]
        print("Preds", predictions)
        print("True values: ",Y)
        ### END CODE
        return out

    def predict(self, X):
        """ Compute classifier prediction on each data point in X 

        Args:
           X: numpy array shape (n, d) - the data each row is a data point
        Returns
           out: np.array shape (n, ) - prediction on each data point (number in 0,1,..., num_classes
        """
        out = np.zeros(X.shape[0])
        ### YOUR CODE HERE - 1-4 lines

        z = softmax(X @ self.W)
        out = np.array([np.argmax(x) for x in z])
        #out = np.dot(X, self.W)
        ### END CODE
        return out
    


def test_encoding():
    print('*'*10, 'Test one-in-k Encoding\n')
    labels = np.array([0, 2, 1, 1])
    m = one_in_k_encoding(labels, 3)
    res =  np.array([[1, 0 , 0], [0, 0, 1], [0, 1, 0], [0, 1, 0]])
    assert res.shape == m.shape, 'encoding shape mismatch'
    assert np.allclose(m, res), m - res
    print('Test Passed')

    
def test_softmax():
    print('*'*5, 'Test softmax\n')
    X = np.zeros((3 ,2))
    X[0, 0] = np.log(4)
    X[1, 1] = np.log(2)
    print('Input to Softmax: \n', X)
    sm = softmax(X)
    expected = np.array([[4.0/5.0, 1.0/5.0], [1.0/3.0, 2.0/3.0], [0.5, 0.5]])
    print('Result of softmax: \n', sm)
    assert np.allclose(expected, sm), 'Expected {0} - got {1}'.format(expected, sm)
    print('Test complete')
        

def test_grad():
    print('*'*5, 'Testing  Gradient\n')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])    
    w = np.ones((2, 3))
    y = np.array([0, 1, 2])
    scl = SoftmaxClassifier(num_classes=3)
    f = lambda z: scl.cost_grad(X, y, W=z)
    numerical_grad_check(f, w)
    print('Test Success')

def test_fit():
    print('*'*5, 'Testing  fit\n')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0,3.0], [1.0, 3.0]])
    y = np.array([0, 2, 1, 2])
    lr = SoftmaxClassifier(num_classes=3)
    lr.fit(X=X, Y=y, epochs=1000, batch_size=2)
    print("Weights", lr.W)
    print("Score", lr.score(X, y))
    print('Test Success')

    
if __name__ == "__main__":
    test_encoding()
    test_softmax()
    test_grad()
    test_fit()
