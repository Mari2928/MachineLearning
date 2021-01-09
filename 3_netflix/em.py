"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from numpy import reshape
from scipy.integrate._ivp.radau import MU_REAL


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    mu, var, pi = mixture 
    
    # identity matrix: 1 for rated movie and 0 otherwise
    delta = X.astype(bool).astype(int) 
    
    # log(N(x; u, var)) with rate information
    f = (np.sum(X**2, axis=1)[:,None] + (delta @ mu.T**2) - 2*(X @ mu.T)) / (2*var)
    pre_exp = (-np.sum(delta, axis=1).reshape(-1,1)/2.0) @ (np.log((2*np.pi*var)).reshape(-1,1)).T
    f = pre_exp - f
    
    # log(pi * N(x; u, var))
    f = f + np.log(pi + 1e-16)
    
    # log of normalizing term in p(j|u)
    logsums = logsumexp(f, axis=1).reshape(-1,1)  # Store this to calculate log_lh
    log_posts = f - logsums # log of posterior prob. matrix: log(p(j|u))    
    log_likelihood = np.sum(logsums)
    post = np.exp(log_posts)

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    _, K = post.shape
    mu_rev, _, _ = mixture 
    
    delta = X.astype(bool).astype(int)
    
    # update pis
    p_rev = post.sum(axis=0) / n
    
    # update means only when sum_u(post * delta) >= 1
    denom = post.T @ delta 
    numer = post.T @ X 
    update_indices = np.where(denom >= 1)
    mu_rev[update_indices] = numer[update_indices] / denom[update_indices]
    
    # update variances
    denom_var = np.sum(post * np.sum(delta, axis=1).reshape(-1,1), axis=0)    
    norms = np.sum(X**2, axis=1)[:,None] + (delta @ mu_rev.T**2) - 2*(X @ mu_rev.T)    
    var_rev = np.maximum(np.sum(post*norms, axis=0) / denom_var, min_variance)

    return GaussianMixture(mu_rev, var_rev, p_rev)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_cost = None
    cost = None
    while (prev_cost is None or abs(cost - prev_cost) >= 1e-6 * abs(cost)):
        prev_cost = cost
        post, cost = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, cost


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X_pred = X.copy()
    mu, var, pi = mixture 
    
    miss_indices = np.where(X == 0)
    delta = X.astype(bool).astype(int) 
    
    # log(N(x; u, var)) with rate information
    f = (np.sum(X**2, axis=1)[:,None] + (delta @ mu.T**2) - 2*(X @ mu.T)) / (2*var)
    pre_exp = (-np.sum(delta, axis=1).reshape(-1,1)/2.0) @ (np.log((2*np.pi*var)).reshape(-1,1)).T
    f = pre_exp - f
    # log(pi * N(x; u, var))
    f = f + np.log(pi + 1e-16)
    
    # log of normalizing term in p(j|u)
    logsums = logsumexp(f, axis=1).reshape(-1,1)
    log_posts = f - logsums # log of posterior prob. matrix: log(p(j|u))    
    post = np.exp(log_posts)
    
    X_pred[miss_indices] = (post @ mu)[miss_indices]  

    return X_pred
