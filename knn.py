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
        x: vector in Rp
        y: collection of vectors in Rp
        p: norm in which we are interested
    """
    distanz = []

    for k in range(np.shape(y)[0]):
        distanz.append((np.sum(x-y[k])**p)**(1./p))

    return np.array(distanz)

def most_common(l):
    """
        Most common element in list
    """
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
    
    def knn_prediction(self, x_test):
        """
            Method knn_prediction: performs a prediction to select the most
            common neighbour of a point.
            -------------------------------------------------------------------
            Input: x_test, set of data in which we are interested on making
            predictions.
        """
        neighbour_list = []

        """
            First loop: append to a list all labels based on
            sorted distances.
        """

        for x in x_test:
            distance = self.metric(x, self.x_train) 
            for i, j in sorted(zip(distance, self.y_train)):
                neighbour_list.append(j)
        
        neighbour = []

        """
            Second loop: for each element in the list take the most
            common neighbour (cut when you count k elements)
        """
        neighbour_list = np.array(neighbour_list)
        neighbour_list = neighbour_list.reshape(np.shape(x_test)[0],np.shape(self.x_train)[0])


        for element in neighbour_list:
            neighbour.append(most_common(list(element[:self.k])))
        
        return np.array(neighbour)

    def loss_function(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        """
            Method loss_function: compute the binary 0/1 loss function
            of the test data.
            -----------------------------------------------------------
            Input: test features (x_test) and test labels (y_test).
        """
        y_predicted = self.knn_prediction(self.x_test)
        loss = np.mean(y_predicted == self.y_test)
        return loss

# We can start with small data: important notes explained below
# First column corresponds to labels, rest of columns correspond to features

train_data_small = np.array(pd.read_csv('Training data/MNIST_train_small.csv'))
test_data_small = np.array(pd.read_csv('Test data/MNIST_test_small.csv'))

y_train, x_train = train_data_small[:,0], train_data_small[:,1:]
y_test, x_test = test_data_small[:,0], test_data_small[:,1:]

K = np.arange(1,40,1)
acc = []
for k in K:
    method = kNN(k)
    method.training_data(x_train, y_train)
    method.knn_prediction(x_test)
    acc.append(method.loss_function(x_test, y_test))

plt.figure()
plt.plot(K, acc, label = 'Accuracy of kNN')
plt.legend()
plt.show()
