import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    correct_predictions = np.sum(y_pred == y_true)
    total_samples = len(y_true)
    
    accuracy = correct_predictions / total_samples

    return accuracy

def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    correct_predictions = np.sum(y_pred == y_true)
    total_samples = len(y_true)
    
    accuracy = correct_predictions / total_samples

    return accuracy

def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    mean_y_true = np.mean(y_true)
    tot = np.sum((y_true - mean_y_true) ** 2)
    res = np.sum((y_true - y_pred) ** 2)

    r2 = 1 - res / tot if tot > 0 else 0

    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    squared_errors = (y_pred - y_true) ** 2
    mse = np.mean(squared_errors)
    
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    absolute_errors = np.abs(y_pred - y_true)
    mae = np.mean(absolute_errors)
    
    return mae