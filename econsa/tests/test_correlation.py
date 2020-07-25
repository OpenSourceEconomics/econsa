"""Test for correlation transformation."""
import chaospy as cp
import numpy as np
import pytest
from scipy.stats import norm

from econsa.correlation import gc_correlation


def get_strategies(name):
    dim = np.random.randint(2, 5)
    means = np.random.uniform(-100, 100, dim)

    if name == "test_gc_correlation":
        # list of distributions to draw from
        # repeated distributions are for higher drawn frequency, not typo
        distributions = [
            cp.Normal,
            cp.Uniform,
            cp.Uniform,
            cp.LogNormal,
            cp.LogNormal,
            cp.Exponential,
            cp.Rayleigh,
            cp.LogWeibull,
        ]

        marginals = list()
        for mean in means:
            dist_i = np.random.choice(len(distributions))
            dist = distributions[dist_i]
            marginals.append(dist(mean))

        # redraw corr until is positive definite
        while True:
            corr = np.random.uniform(-1, 1, size=(dim, dim))
            corr = corr @ corr.T
            for i in range(dim):
                corr[i, i] = 1
            if np.all(np.linalg.eigvals(corr) > 0) == 1:
                break
    elif name == "test_gc_correlation_exception_marginals":
        marginals = list()
        for i in range(dim):
            marginals.append(norm())

        corr = np.random.uniform(-1, 1, size=(dim, dim))
    elif name == "test_gc_correlation_exception_corr_symmetric":
        distributions = [cp.Normal, cp.Uniform, cp.LogNormal]
        marginals = list()
        for mean in means:
            dist_i = np.random.choice(len(distributions))
            dist = distributions[dist_i]
            marginals.append(dist(mean))

        corr = np.random.uniform(-1, 1, size=(dim, dim))
    else:
        raise NotImplementedError

    strategy = (marginals, corr)
    return strategy


def test_gc_correlation():
    marginals, corr = get_strategies("test_gc_correlation")

    # re-calc corr until is positive definite
    while True:
        corr_transformed = gc_correlation(marginals, corr)
        if np.all(np.linalg.eigvals(corr_transformed) > 0) == 1:
            break

    copula = cp.Nataf(cp.J(*marginals), corr_transformed)
    corr_copula = np.corrcoef(copula.sample(1000000))

    np.testing.assert_almost_equal(corr, corr_copula, decimal=1)


def test_gc_correlation_exception_marginals():
    marginals, corr = get_strategies("test_gc_correlation_exception_marginals")

    with pytest.raises(NotImplementedError) as e:
        gc_correlation(marginals, corr)
    assert "marginals must be chaospy distributions" in str(e.value)


def test_gc_correlation_exception_corr_symmetric():
    marginals, corr = get_strategies("test_gc_correlation_exception_corr_symmetric")

    with pytest.raises(ValueError) as e:
        gc_correlation(marginals, corr)
    assert "corr is not a symmetric matrix" in str(e.value)
