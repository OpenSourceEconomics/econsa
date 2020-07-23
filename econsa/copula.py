"""Conditional sampling from Gaussian copula.

This module contains functions to sample random input parameters from a Gaussian copula.
"""
import numpy as np
from scipy.stats import multivariate_normal as multivariate_norm
from scipy.stats import norm

from econsa.sampling import cond_mvn


# def cond_gaussian_copula_sample(cov, dependent_ind, given_ind, given_value_u, distribution):
#     """Summary.

#     .. todo::
#         The code now only takes one given value.

#     Returns
#     -------
#     TYPE
#         Description.
#     """
#     condi_value_u = cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u)

#     gc_value = distribution[int(dependent_ind[0])].inv(condi_value_u)
#     return gc_value


def cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u):
    r"""Conditional sampling from Gaussian copula.

    This function provides the probability distribution of conditional sample
    drawn from a Gaussian copula, given covariance matrix and a uniform random vector,
    as described in Section 4.1 of [K2012]_.

    .. todo::
        Make this the backend of ``gc_value``.

    Parameters
    ----------
    cov : array_like
        Covariance matrix of the desired sample.

    dependent_ind : int or array_like
        The indices of dependent variables.

    given_ind : array_like
        The indices of independent variables.

    given_value_u : array_like
        The given random vector (:math:`u`) that is uniformly distributed between 0 and 1.

    Returns
    -------
    cond_quan : array_like
        The conditional sample (:math:`G(u)`) that is between 0 and 1,
        and has the same length as ``dependent_ind``.

    References
    ----------
    .. [K2012] Kucherenko, S., Tarantola, S., & Annoni, P. (2012).
        Estimation of global sensitivity indices for models with
        dependent variables. Computer Physics Communications, 183(4), 937–946.

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

    # Check `given_value_u` are between 0 and 1
    if not np.all((given_value_u >= 0) & (given_value_u <= 1)):
        raise ValueError("given_value_u must be between 0 and 1")

    # Make sure that covariance is well-conditioned
    if np.linalg.cond(cov) > 100:
        raise ValueError("covariance matrix is ill-conditioned")

    # F^{−1}(u)
    given_value_y = norm().ppf(given_value_u)

    means = np.zeros(cov.shape[0])
    cond_mean, cond_cov = cond_mvn(
        means, _cov2corr(cov), dependent_ind, given_ind, given_value_y,
    )

    # C(u, Sigma)
    cond_dist = multivariate_norm(cond_mean, cond_cov)
    cond_draw = np.atleast_1d(cond_dist.rvs())
    # Conditional F^{-1}(u)
    cond_quan = np.atleast_1d(norm.cdf(cond_draw))

    return cond_quan


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
         If True then the standard deviation is also returned. (default value is `False`)

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
