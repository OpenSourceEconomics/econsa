"""Test for correlation transformation."""
import chaospy as cp
import numpy as np
import pytest
from scipy.stats import norm
from statsmodels.stats.correlation_tools import corr_nearest

from econsa.copula import _cov2corr
from econsa.correlation import gc_correlation


def get_strategies(name):
    dim = np.random.randint(2, 10)
    means = np.random.uniform(-100, 100, dim)

    if name == "test_gc_correlation_functioning":
        # List of distributions to draw from.
        # Repeated distributions are for higher drawn frequency, not typo
        distributions = [
            cp.Exponential,
            cp.Gilbrat,
            cp.HyperbolicSecant,
            cp.Laplace,
            cp.LogNormal,
            cp.LogNormal,
            cp.LogNormal,
            cp.LogUniform,
            cp.LogUniform,
            cp.LogUniform,
            cp.LogWeibull,
            cp.Logistic,
            cp.Maxwell,
            cp.Normal,
            cp.Normal,
            cp.Normal,
            cp.Rayleigh,
            cp.Uniform,
            cp.Uniform,
            cp.Uniform,
            cp.Wigner,
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
    elif (
        name == "test_gc_correlation_2d" or name == "test_gc_correlation_2d_force_calc"
    ):
        dim = 2
        means = np.random.uniform(-100, 100, dim)
        distributions = [
            cp.Normal,
            cp.Uniform,
            cp.Exponential,
            cp.Rayleigh,
            cp.LogWeibull,
        ]
        marginals = [cp.Normal(means[0])]
        dist2 = distributions[np.random.choice(len(distributions))](means[1])
        marginals.append(dist2)

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
    """Test the function runs successfully."""
    marginals, corr = get_strategies("test_gc_correlation_functioning")
    corr_transformed = gc_correlation(marginals, corr)
    cp.Nataf(cp.J(*marginals), corr_transformed)
    return "the function ended without error"


def test_gc_correlation_2d():
    """Test for special combinations the results are accurate."""
    marginals, corr = get_strategies("test_gc_correlation_2d")
    corr_transformed = gc_correlation(marginals, corr)
    copula = cp.Nataf(cp.J(*marginals), corr_transformed)
    corr_copula = np.corrcoef(copula.sample(10000000))
    np.testing.assert_almost_equal(corr, corr_copula, decimal=3)


@pytest.mark.xfail
def test_gc_correlation_2d_force_calc():
    """Test the results from force_calc are close to that from the paper."""
    marginals, corr = get_strategies("test_gc_correlation_2d_force_calc")
    corr_ref_numbers = gc_correlation(marginals, corr)
    corr_force_calc = gc_correlation(marginals, corr, force_calc=True)
    assert np.all(np.absolute(corr_ref_numbers - corr_force_calc) <= 0.3) == 1


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
