"""Conditional sampling from Gaussian copula.

This module contains functions to sample random input parameters from a Gaussian copula.
"""
import numpy as np
from scipy.stats import multivariate_normal as multivariate_norm
from scipy.stats import norm

from econsa.sampling import cond_mvn


def cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u):
    """Correlation for Gaussian copula.

    Parameters
    ----------
    cov : array_like
        Description

    dependent_ind : int or array_like
        The indices of dependent variables.

    given_ind : array_like
        The indices of independent variables.

    given_value_u : TYPE
        Description

    Returns
    -------
    cond_quan : array_like
        Description

    Examples
    --------
    >>> import chaospy as cp
    >>> np.random.seed(123)
    >>> dim = 3
    >>> means = np.random.uniform(-100, 100, dim)
    >>> sigma = np.random.normal(size=(dim, dim))
    >>> cov = sigma @ sigma.T
    >>> marginals = list()
    >>> for i in range(dim):
    ...     mean, sigma = means[i], np.sqrt(cov[i, i])
    ...     marginals.append(cp.Normal(mu=mean, sigma=sigma))
    >>> distribution = cp.J(*marginals)
    >>> sample = distribution.sample(1).T[0]
    >>> full = list(range(0, dim))
    >>> dependent_ind = [np.random.choice(full)]
    >>> given_ind = full[:]
    >>> given_ind.remove(dependent_ind[0])
    >>> given_value = sample[given_ind]
    >>> given_value_u = [
    ...     distribution[ind].cdf(given_value[i]) for i, ind in enumerate(given_ind)
    ... ]
    >>> condi_value_u = cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u)
    >>> np.testing.assert_almost_equal(condi_value_u[0], 0.170718, decimal=6)
    """
    given_value_u = np.atleast_1d(given_value_u)

    # Check `given_value_u` are between 0 and 1:
    if not np.all((given_value_u >= 0) & (given_value_u <= 1)):
        raise ValueError("sanitize your inputs!")

    given_value_y = norm().ppf(given_value_u)

    means = np.zeros(cov.shape[0])
    cond_mean, cond_cov = cond_mvn(
        means, _cov2corr(cov), dependent_ind, given_ind, given_value_y,
    )

    cond_dist = multivariate_norm(cond_mean, cond_cov)
    cond_draw = np.atleast_1d(cond_dist.rvs())
    cond_quan = norm.cdf(cond_draw)

    return np.atleast_1d(cond_quan)


def _cov2corr(cov, return_std=False):
    r"""Convert covariance matrix to correlation matrix.

    This function does not convert subclasses of `ndarray`s.
    This requires that division is defined element-wise.
    `np.ma.array` and `np.matrix` are allowed.

    Parameters
    ----------
    cov: array_like
         Covariance matrix with dimensions :math:`N\times N`.

    return_std: bool, optional
         If True then the standard deviation is also returned.
         By default only the correlation matrix is returned.

    Returns
    -------
    corr: ndarray
        Correlation matrix with dimensions :math:`N\times N`.

    std_: ndarray
        Standard deviation with dimensions :math:`1\times N`.
    """
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr
