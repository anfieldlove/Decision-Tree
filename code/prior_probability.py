import numpy as np

class PriorProbability():
    def __init__(self):
        """
        This is a simple classifier that only uses prior probability to classify 
        points. It just looks at the classes for each data point and always predicts
        the most common class.

        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Implement a classifier that works by prior probability. Takes in features
        and targets and fits the features to the targets using prior probability.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N 
                examples.
        """

        if np.count_nonzero(targets == 0) > np.count_nonzero(targets == 1):
            predict_as = 0
        else:
            predict_as = 1

        self.most_common_class = predict_as


    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """

        targets_value = []
        length = len(data)
        if self.most_common_class == 0:
            targets_value = np.zeros(length, int)
        else:
            targets_value = np.ones(length, int)

        return np.array(targets_value)