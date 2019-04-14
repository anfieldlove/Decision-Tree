import numpy as np

def confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the confusion matrix. The confusion 
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    YOU DO NOT NEED TO IMPLEMENT CONFUSION MATRICES THAT ARE FOR MORE THAN TWO 
    CLASSES (binary).
    
    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """
    '''if len(predictions) == len(actual):
        print("True")
    else:
        print("False")'''
    #print("llllll=", len(predictions))
    #print("sssssss=", len(actual))
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    predict_true_index = np.where(predictions == 1)
    predict_false_index = np.where(predictions == 0)

    true_positives = np.count_nonzero(actual[predict_true_index] == 1)
    false_positives = np.count_nonzero(actual[predict_true_index] == 0)
    true_negatives = np.count_nonzero(actual[predict_false_index] == 0)
    false_negatives = np.count_nonzero(actual[predict_false_index] == 1)

    confusion_matrix = np.array([[true_negatives, false_positives],
                                 [false_negatives, true_positives]])

    return confusion_matrix

def accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the accuracy:

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    num_correct = confusion_matrix(actual, predictions)[0][0] + confusion_matrix(actual, predictions)[1][1]
    num_total = np.sum(confusion_matrix(actual, predictions))

    accuracy = num_correct / num_total

    return accuracy

def precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    Hint: implement and use the confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    num_positive = confusion_matrix(actual, predictions)[0][1] + confusion_matrix(actual, predictions)[1][1]
    total_positive = confusion_matrix(actual, predictions)[1][1] + confusion_matrix(actual, predictions)[1][0]
    precision = confusion_matrix(actual, predictions)[1][1] \
                / num_positive
    recall = confusion_matrix(actual, predictions)[1][1] \
             / total_positive

    return precision, recall

def f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length 
    numpy vector), compute the F1-measure:

   https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Hint: implement and use the precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and 
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    P, R = precision_and_recall(actual, predictions)
    return 2 * P * R / (P + R)

