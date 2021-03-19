import numpy as np

def count_true(array_binary):
    return np.sum(array_binary.astype('uint8'))

def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy

    accuracy = count_true(prediction==ground_truth)/ground_truth.shape[0]
    return accuracy
