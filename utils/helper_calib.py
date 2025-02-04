import numpy as np

def stochastic_simplex_sampling(K, N, alpha=1.0):
    """
    Stochastically sample points from a (K-1)-dimensional simplex.

    Args:
        K (int): Number of dimensions (e.g., 4000 classes).
        N (int): Number of samples to generate.
        alpha (float): Dirichlet concentration parameter (alpha=1 gives uniform sampling).

    Returns:
        np.ndarray: Array of shape (N, K) where each row is a valid probability vector.
    """
    return np.random.dirichlet(alpha=np.ones(K) * alpha, size=N)