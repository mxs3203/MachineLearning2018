import numpy as np
import h1_util
from sklearn.utils import shuffle
import scipy
from sklearn.utils import resample

def logistic(z):
    
    
    """ 
    Helper function
    Computes the logistic function 1/(1+e^{-x}) to each entry in input vector z.
    
    np.exp may come in handy
    Args:
        z: numpy array shape (d,) 
    Returns:
       logi: numpy array shape (d,) each entry transformed by the logistic function 
    """
    logi = np.zeros(z.shape)
    ### YOUR CODE HERE 1-5 lines
    logi = np.array([1/(1+np.exp(-i)) for i in z])
    ### END CODE
    assert logi.shape == z.shape
    return logi

#


class LogisticRegressionClassifier():

    w = []
    history = []

    def __init__(self):
        self.w = None

    def cost_grad(self, X, y, w):
        """
        Compute the average cross entropy and the gradient under the logistic regression model 
        using data X, targets y, weight vector w 
        
        np.log, np.sum, np.choose, np.dot may be useful here
        Args:
           X: np.array shape (n,d) float - Features 
           y: np.array shape (n,)  int - Labels 
           w: np.array shape (d,)  float - Initial parameter vector

        Returns:
           cost: scalar the cross entropy cost of logistic regression with data X,y 
           grad: np.arrray shape(n,d) gradient of cost at w 
        """
        cost = 0
        grad = np.zeros(w.shape)
        ### YOUR CODE HERE 5 - 15 lines
        
        cost = -np.sum( y*np.log(logistic(X.dot(np.transpose(w)))) + (1-y)*np.log(1-logistic(X.dot(np.transpose(w)))) )
        cost = 1.0/len(X) * cost

        for i in range(len(grad)):
            # slide 51
            deltaNLL = y - logistic(X.dot(w.transpose())) 
            deltaNLL = -deltaNLL.transpose().dot(X[:, i])
            grad[i] = (1.0 / len(X)) * deltaNLL

        ### END CODE
        assert grad.shape == w.shape
        return cost, grad


    def fit(self, X, y, w=None, lr=0.1, batch_size=3, epochs=10):
        """
        Run mini-batch stochastic Gradient Descent for logistic regression 
        use batch_size data points to compute gradient in each step.
    
        The function np.random.permutation may prove useful for shuffling the data before each epoch
        It is wise to print the performance of your algorithm at least after every epoch to see if progress is being made.
        Remeber the stochastic nature of the algorithm may give fluctuations in the cost as iterations increase.

        Args:
           X: np.array shape (n,d) dtype float32 - Features 
           y: np.array shape (n,) dtype int32 - Labels 
           w: np.array shape (d,) dtype float32 - Initial parameter vector
           lr: scalar - learning rate for gradient descent
           batch_size: number of elements to use in minibatch
           epochs: Number of scans through the data

        sets: 
           w: numpy array shape (d,) learned weight vector w
           history: list/np.array len epochs - value of cost function after every epoch. You know for plotting
        """
        if w is None: w = np.zeros(X.shape[1])
        history = []
        n = X.shape[0]
        ### YOUR CODE HERE 14 - 20 lines
        for i in range(epochs):
            X_shuffle,Y_shuffle = shuffle(X, y)
            for j in range(n // batch_size):
                X_subset = resample(X_shuffle, n_samples = batch_size, random_state=0)
                y_subset = resample(Y_shuffle, n_samples = batch_size, random_state=0)
                cost, grad = self.cost_grad(X_subset, y_subset, w)  # compute cost and gradient of the data with weights
                history.append(cost)  # remember the loss in iteration
                w -= lr * grad  # upgrade weights depending on gradient
                #lr = lr * 0.99
                print("Cost",cost)
                print("Gradient", grad)

        ### END CODE
        self.w = w
        self.history = history

    def fast_descent(self,X, y, w=None, reg=0, rounds=100):
        """ Uses fancy optimizer to do the gradient descent """
        # unstable if linear separable...
        if w is None: w = np.zeros(X.shape[1])
        w = scipy.optimize.minimize(lambda t: self.cost_grad(X, y, t, reg), w, jac=True,
                                    options={'maxiter': rounds, 'disp': True})
        return w.x

    def predict(self, X):
        """ Classify each data element in X

        Args:
            X: np.array shape (n,d) dtype float - Features 
        
        Returns: 
           p: numpy array shape (n, ) dtype int32, class predictions on X (0, 1)

        """
        pred = np.zeros(X.shape[0])
        ### YOUR CODE HERE 1 - 4 lines

        z = logistic(X @ self.w)
        predictions = np.array([0 if x < 0.5 else 1 for x in z])
        ### END CODE
        assert predictions.shape == (X.shape[0],)
        return predictions.astype('int64')
    
    def score(self, X, y):
        """ Compute model accuracy  on Data X with labels y

        Args:
            X: np.array shape (n,d) dtype float - Features 
            y: np.array shape (n,) dtype int32 - Labels 

        Returns: 
           s: float, number of correct prediction divivded by n.

        """
        s = 0
        ### YOUR CODE HERE 1 - 4 lines
        predictions = self.predict(X)
        #print(predictions)
        #print(y)
        #print(X.shape[0])
        acc = np.sum(predictions == y)/X.shape[0]
        ### END CODE
        return acc
        

    
def test_logistic():
    print('*'*5, 'Testing logistic function')
    a = np.array([0, 1, 2, 3])
    lg = logistic(a)
    target = np.array([ 0.5, 0.73105858, 0.88079708, 0.95257413])
    assert np.allclose(lg, target), 'Logistic Mismatch Expected {0} - Got {1}'.format(target, lg)
    print('Test Success!')

    
def test_cost():
    print('*'*5, 'Testing Cost Function')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [3, 2]])
    y = np.array([0, 0, 1], dtype='int64')
    w = np.array([0.0, 0.0])
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    cost,_ = lr.cost_grad(X, y, w)
    target = -np.log(0.5)
    assert np.allclose(cost, target), 'Cost Function Error:  Expected {0} - Got {1}'.format(target, cost)
    print('Test Success')

    
def test_grad():
    print('*'*5, 'Testing  Gradient')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])    
    w = np.array([0.0, 0.0])
    y = np.array([0, 0, 1]).astype('int64')
    print('shapes', X.shape, w.shape, y.shape)
    lr = LogisticRegressionClassifier()
    f = lambda z: lr.cost_grad(X, y, w=z)
    h1_util.numerical_grad_check(f, w)
    print('Test Success')


def test_fit():
    print('*' * 5, 'Testing  Fit')
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    w = np.array([0.0, 0.0])
    y = np.array([0, 0, 1]).astype('int64')
    lr = LogisticRegressionClassifier()
    lr.fit(X=X, y=y, w=w, epochs=10000)
    print("Weights",lr.w)
    print("Score", lr.score(X,y))
    print('Test Success')

    
if __name__ == '__main__':
    test_logistic()
    test_cost()
    test_grad()
    test_fit()
