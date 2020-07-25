"""Test for correlation transformation."""
import chaospy as cp
import numpy as np
import pytest
from scipy.stats import norm
from statsmodels.stats.correlation_tools import corr_nearest

from econsa.copula import _cov2corr
from econsa.correlation import gc_correlation


def get_strategies(name):
    dim = np.random.randint(2, 5)
    means = np.random.uniform(-100, 100, dim)

    if name == "test_gc_correlation_functioning":
        # List of distributions to draw from.
        distributions = [
            cp.Normal,
            cp.Uniform,
            cp.LogNormal,
            cp.Exponential,
            cp.Rayleigh,
            cp.LogWeibull,
        ]

        marginals = list()
        for mean in means:
            dist = distributions[np.random.choice(len(distributions))](mean)
            marginals.append(dist)

        cov = np.random.uniform(-1, 1, size=(dim, dim))
        cov = cov @ cov.T
        # If not positive definite, find the nearest one.
        if np.all(np.linalg.eigvals(cov) > 0) == 0:
            cov = corr_nearest(cov)

        corr = _cov2corr(cov).round(8)
    elif name == "test_gc_correlation_exception_marginals":
        marginals = list()
        for i in range(dim):
            marginals.append(norm())

        corr = np.random.uniform(-1, 1, size=(dim, dim))
    elif name == "test_gc_correlation_exception_corr_symmetric":
        distributions = [cp.Normal, cp.Uniform, cp.LogNormal]
        marginals = list()
        for mean in means:
            dist = distributions[np.random.choice(len(distributions))](mean)
            marginals.append(dist)

        corr = np.random.uniform(-1, 1, size=(dim, dim))
    else:
        raise NotImplementedError

    strategy = (marginals, corr)
    return strategy


def test_gc_correlation_functioning():
    marginals, corr = get_strategies("test_gc_correlation_functioning")
    corr_transformed = gc_correlation(marginals, corr)
    cp.Nataf(cp.J(*marginals), corr_transformed)
    return "the function ended without error"


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
