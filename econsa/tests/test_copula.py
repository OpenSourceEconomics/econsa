"""Tests for copula sampling.

This module contains tests for the copula sampling functions.

"""
import chaospy as cp
import numpy as np
import pytest
from scipy.stats import multivariate_normal as multivariate_norm

from econsa.copula import cond_gaussian_copula
from econsa.sampling import cond_mvn


def get_strategies(name):
    dim = np.random.randint(2, 10)

    full = list(range(0, dim))
    given_ind = full[:]
    dependent_ind = [np.random.choice(full)]
    given_ind.remove(dependent_ind[0])

    means = np.random.uniform(-100, 100, dim)
    sigma = np.random.normal(size=(dim, dim))

    exception_cov = False

    cov = sigma @ sigma.T
    if np.linalg.cond(cov) > 100:
        exception_cov = True

    marginals = list()
    for i in range(dim):
        mean, sigma = means[i], np.sqrt(cov[i, i])
        marginals.append(cp.Normal(mu=mean, sigma=sigma))

    distribution = cp.J(*marginals)

    sample = distribution.sample(1).T[0]
    given_value = sample[given_ind]

    if name == "test_cond_gaussian_copula":
        np.random.seed(123)
        given_value_u = [
            distribution[ind].cdf(given_value[i]) for i, ind in enumerate(given_ind)
        ]

        strategy_gc = (cov, dependent_ind, given_ind, given_value_u, distribution)
        strategy_cn = (means, cov, dependent_ind, given_ind, given_value)
        strategy = (strategy_gc, strategy_cn, exception_cov)
    elif name == "test_cond_gaussian_copula_exception_u":
        given_value_u = given_value
        strategy = (cov, dependent_ind, given_ind, given_value_u)
    else:
        raise NotImplementedError

    return strategy


def test_cond_gaussian_copula():
    """
    The results from a Gaussian copula with normal marginal distributions
    should be identical to the direct use of a multivariate normal
    distribution.
    """
    args_gc, args_cn, exception_cov = get_strategies("test_cond_gaussian_copula")
    cov, dependent_ind, given_ind, given_value_u, distribution = args_gc

    if exception_cov is True:
        # Test error when ``cov`` has large conditional number
        with pytest.raises(ValueError) as e:
            cond_gaussian_copula(cov, dependent_ind, given_ind, given_value_u)
        assert "covariance matrix is ill-conditioned" in str(e.value)
    else:
        # Test valid case
        condi_value_u = cond_gaussian_copula(
            cov, dependent_ind, given_ind, given_value_u,
        )
        gc_value = distribution[int(dependent_ind[0])].inv(condi_value_u)

        np.random.seed(123)
        cond_mean, cond_cov = cond_mvn(*args_cn)
        cond_dist = multivariate_norm(cond_mean, cond_cov)
        cn_value = np.atleast_1d(cond_dist.rvs())

        np.testing.assert_almost_equal(cn_value, gc_value)


def test_cond_gaussian_copula_exception_u():
    """Test cond_gaussian_copula raises exceptions when invalid ``given_value_u`` is passed.
    """
    args = get_strategies("test_cond_gaussian_copula_exception_u")

    with pytest.raises(ValueError) as e:
        cond_gaussian_copula(*args)
    assert "given_value_u must be between 0 and 1" in str(e.value)
