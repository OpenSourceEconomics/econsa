"""Test for correlation transformation."""
import chaospy as cp
import numpy as np

from econsa.correlation import gc_correlation


def get_strategies(name):
    dim = np.random.randint(2, 5)
    means = np.random.uniform(-100, 100, dim)

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
        dist_i = np.random.choice(len(distributions))
        dist = distributions[dist_i]
        marginals.append(dist(mean))

    if name == "test_gc_correlation":
        # redraw corr until is positive definite
        while True:
            corr = np.random.uniform(-1, 1, size=(dim, dim))
            corr = corr @ corr.T
            for i in range(dim):
                corr[i, i] = 1
            if np.all(np.linalg.eigvals(corr) > 0) == 1:
                break

        strategy = (marginals, corr)
    else:
        raise NotImplementedError

    return strategy


def test_gc_correlation():
    marginals, corr = get_strategies("test_gc_correlation")

    # re-calc corr until is positive definite
    while True:
        corr_transformed = gc_correlation(marginals, corr)
        if np.all(np.linalg.eigvals(corr_transformed) > 0) == 1:
            break

    copula = cp.Nataf(cp.J(*marginals), corr_transformed)
    corr_copula = np.corrcoef(copula.sample(1000000)).round(3)

    np.testing.assert_almost_equal(corr, corr_copula, decimal=1)


def test_gc_correlation_exception():
    # TODO
    pass
