
import numpy as np


def binary_cross_entropy(y_true, y_prob, eps=1e-15):
    """
    Calculate binary cross-entropy loss.

    Parameters:
        y_true: Actual binary labels, shape (N,)
        y_prob: Predicted probabilities, shape (N,)
        eps: Small constant to prevent log(0)

    Returns:
        Mean binary cross-entropy.
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)

    return -np.mean(
        y_true * np.log(y_prob)
        + (1 - y_true) * np.log(1 - y_prob)
    )


def accuracy(y_true, y_pred):
    """Calculate classification accuracy."""
    return np.mean(y_true == y_pred)


def classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Calculate basic binary classification metrics.

    Returns:
        Dictionary containing cross-entropy and accuracy.
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "cross_entropy": binary_cross_entropy(y_true, y_prob),
        "accuracy": accuracy(y_true, y_pred)
    }
