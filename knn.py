import numpy as np 
import matplotlib.pyplot as plt 

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
    return (np.sum(x-y, axis = 1)**p)**(1./p) # returns all distances from x to vector y

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

        for element in neighbour_list:
            neighbour.append(most_common(element[:self.k]))
        
        return neighbour

    def loss_function(self, x_test, y_test):
        """
            Method loss_function: compute the binary 0/1 loss function
            of the test data.
            -----------------------------------------------------------
            Input: test features (x_test) and test labels (y_test).
        """
        y_predicted = knn_prediction(x_test)
        loss = np.mean(y_predicted == y_test)
        return loss





    

