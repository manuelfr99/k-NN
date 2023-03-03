import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

"""
    Defining a k-NN algorithm from scratch. 
    Idea: we have a dataset of points (Xi, Yi) and based on the
    features we want to make classification of data into different 
    classes.
"""

def euclidean_distance(x, y, p = 2):
    """                                     
    Inputs:
        x,y: vectors Rp
        p: norm in which we are interested
    """
    return (np.sum((x-y)**p, axis = 1))**(1./p)

def most_common(l):
    """
        Most common element in list, is this computationally efficient?
    """
    # This approach chooses the smallest label of the repeated in case there are
    # any ties
    return max(set(l), key=l.count)


class kNN:
    """
        Class kNN: this is going to implement a kNN classification system
        into a dataset of points that we previously have divided into training
        and testing.
    """
    def __init__(self, k, metric = euclidean_distance):
        """
            Constructor: takes as inputs k (we can fix a default value) and
            the euclidean_distance, which we have adapted to take any value of
            p we would like.
            -------------------------------------------------------------------
            Inputs: 
            - k: number of neighbours that we are going to use in the 
                 classification
            - metric: by default the euclidean distance previously defined
        """
        self.k = k
        self.metric = euclidean_distance
    
    def training_data(self, x_train, y_train):
        """
            Method training data: takes x_train and y_train and defines them as
            self variables.
            --------------------------------------------------------------------
            Inputs:
            - x_train: training features
            - y_train: training labels
        """
        self.x_train = x_train
        self.y_train = y_train
    
    def knn_prediction(self, x_test, y_test):
        """
            Method knn_prediction: performs a prediction to select the most
            common neighbour of a point.
            -------------------------------------------------------------------
            Input: x_test, set of data in which we are interested on making
            predictions.
        """
        neighbour = []
        
        for x in x_test:
            distance = np.linalg.norm(x-x_train, axis = 1) # only valid if p = 2
            sorted_indices = np.argpartition(distance, self.k)[:self.k+1]
            labels = y_train[sorted_indices]
            neighbour.append(most_common(list(labels)))
        
        return neighbour

    def loss_function_test(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        """
            Method loss_function: compute the binary 0/1 loss function
            of the test data.
            -----------------------------------------------------------
            Input: test features (x_test) and test labels (y_test).
        """
        y_predicted = self.knn_prediction(self.x_test, self.y_test)
        loss = np.mean(y_predicted != self.y_test)
        return loss
    
    def loss_function_train(self, x_train, y_train):
        """
            Computation of the loss function on training data
        """
        y_pred = self.knn_prediction(self.x_train, self.y_test)
        loss = np.mean(y_pred != self.y_train)
        return loss

# We can start with small data: important notes explained below
# First column corresponds to labels, rest of columns correspond to features

train_data_small = np.array(pd.read_csv('Training data/MNIST_train_small.csv'))
test_data_small = np.array(pd.read_csv('Test data/MNIST_test_small.csv'))

y_train, x_train = train_data_small[:,0], train_data_small[:,1:]
y_test, x_test = test_data_small[:,0], test_data_small[:,1:]

K = np.arange(1,20+1,1)
acc_test = []
acc_train = []

from time import time

for k in K:

    start = time()
    method = kNN(k)
    method.training_data(x_train, y_train)
    method.knn_prediction(x_test, y_test)
    acc_test.append(method.loss_function_test(x_test, y_test))
    acc_train.append(method.loss_function_train(x_train, y_train))
    end = time()
    print('Ellapsed time t = ', end - start, ' s')

plt.figure()
plt.plot(K, acc_test, label = 'Empirical Risk of kNN test data')
plt.plot(K, acc_train, label = 'Empirical Risk of kNN train data')
plt.grid()
plt.xlabel('Number of k neighbours')
plt.ylabel('Empirical Risk $R(\\hat{f}_n)$')
plt.legend()
plt.show()
