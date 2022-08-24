import numpy as np


def create_sequences(values, lookback=None, inference=False):
    """
    Create sequences of data.

    Parameters
    ----------
    values : numpy.ndarray
        Array of values.
    lookback : int, optional
        Number of previous values to use as input.
    inference : bool, optional (default=False)  
        Whether to create sequences for inference.

    Returns
    -------
    numpy.ndarray
        Array of sequences.
    numpy.ndarray
        Array of targets.
    """
    if lookback is None:
        lookback = len(values)
    X, Y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i - lookback:i])
        Y.append(values[i])
    if inference:
        return np.stack(X)
    return np.stack(X), np.stack(Y)