import numpy as np


def normalization_range(data):
    mean = data.mean()
    min = data.min()
    max = data.max()
    normalized_data = (data - mean) / (max - min)
    return normalized_data

def normalization_zscore(data):
    mean = data.mean()
    std = data.std()
    z_scores = (data - mean) / std
    return z_scores

def caulate_eingenvalues_eingenvectors(x_norm):
    """
    Calculate the eigenvalues and eigenvectors of the covariance matrix of the normalized data.
    Args:
        x_norm (numpy.ndarray): Normalized data.
    Returns:
        e (numpy.ndarray): Eigenvalues of the covariance matrix.
        v (numpy.ndarray): Eigenvectors of the covariance matrix.
    """
    # Covariance matrix
    covmatrix = np.cov(x_norm.T)

    # Obtain the eingenvalues and eingenvectors of covariance matrix
    e, v = np.linalg.eig(covmatrix)

    # order descendingly by largest eigenvalue
    order = np.argsort((np.argsort(e) * -1))
    e = e[order]
    v = v[:,order]
    return e, v