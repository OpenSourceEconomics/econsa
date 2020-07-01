"""Tests for the sampling.

This module contains all tests for the sampling setup.

"""
import numpy as np
import pytest
import rpy2.robjects.packages as rpackages
from numpy.random import default_rng
from rpy2 import robjects
from rpy2.robjects import numpy2ri

from econsa.sampling import cond_mvn


# Import R modules
r_base = rpackages.importr("base")
r_stats = rpackages.importr("stats")
r_utils = rpackages.importr("utils")
r_utils.chooseCRANmirror(ind=1)
r_utils.install_packages("condMVNorm")
r_cond_mvnorm = rpackages.importr("condMVNorm")

# Import numpy.random generator
rng = default_rng()


def get_strategies(name):
    n = rng.integers(low=4, high=21)
    mean = rng.integers(low=-2, high=2, size=n)
    dependent_n = rng.integers(low=1, high=n - 2)
    dependent = rng.choice(range(0, n), replace=False, size=dependent_n)

    if name == "cond_mvn":
        sigma = rng.standard_normal(size=(n, n))
        sigma = sigma @ sigma.T
        given_ind = [x for x in range(0, n) if x not in dependent]
        given_value = rng.integers(low=-2, high=2, size=len(given_ind))
        strategy = (n, mean, sigma, dependent, given_ind, given_value)
    elif name == "cond_mvn_exception_given":
        sigma = rng.standard_normal(size=(n, n))
        sigma = sigma @ sigma.T
        given_ind = (
            [x for x in range(0, n) if x not in dependent] if n % 3 == 0 else None
        )
        given_value = (
            rng.integers(low=-2, high=2, size=n - dependent_n + 1)
            if n % 2 == 0
            else None
        )
        strategy = (n, mean, sigma, dependent, given_ind, given_value)
    elif name == "test_cond_mvn_exception_sigma":
        sigma = (
            rng.standard_normal(size=(n, n)) if n % 3 == 0 else np.diagflat([-1] * n)
        )
        sigma = sigma @ sigma.T if n % 2 == 0 else sigma
        given_ind = [x for x in range(0, n) if x not in dependent]
        given_value = rng.integers(low=-2, high=2, size=len(given_ind))
        strategy = (n, mean, sigma, dependent, given_ind, given_value)
    else:
        raise NotImplementedError
    return strategy


def test_cond_mvn():
    """Test cond_mvn against the original from R package cond_mvnorm.
    """
    # Evaluate Python code
    n, mean, sigma, dependent_ind, given_ind, given_value = get_strategies("cond_mvn")
    cond_mean, cond_var = cond_mvn(mean, sigma, dependent_ind, given_ind, given_value)

    # Evaluate R code
    numpy2ri.activate()
    r_mean = robjects.FloatVector(mean)
    r_sigma = r_base.matrix(sigma, n, n)
    r_dependent_ind = robjects.IntVector([x + 1 for x in dependent_ind])
    r_given_ind = robjects.IntVector([x + 1 for x in given_ind])
    r_given_value = robjects.IntVector(given_value)
    r_cond_mean, r_cond_var = r_cond_mvnorm.condMVN(
        mean=r_mean,
        sigma=r_sigma,
        dependent=r_dependent_ind,
        given=r_given_ind,
        X=r_given_value,
    )

    r_cond_mean = np.array(r_cond_mean)
    r_cond_var = np.array(r_cond_var)

    numpy2ri.deactivate()

    # Comparison
    np.testing.assert_allclose(cond_mean, r_cond_mean)
    np.testing.assert_allclose(cond_var, r_cond_var)


def test_cond_mvn_exception_given():
    """Test cond_mvn raises exceptions when invalid `given_ind` or `given_value` is passed.
    """
    n, mean, sigma, dependent_ind, given_ind, given_value = get_strategies(
        "cond_mvn_exception_given",
    )

    if n % 3 != 0:
        # Valid case: only `given_ind` is empty
        # or both `given_ind` and `given_value` are empty
        cond_mvn(mean, sigma, dependent_ind, given_ind, given_value)
    else:
        # `given_value` is empty or does not align with `given_ind`
        with pytest.raises(ValueError) as e:
            cond_mvn(mean, sigma, dependent_ind, given_ind, given_value)
        assert "lengths of given_value and given_ind must be the same" in str(e.value)


def test_cond_mvn_exception_sigma():
    """Test cond_mvn raises exceptions when invalid `sigma` is passed.
    """
    n, mean, sigma, dependent_ind, given_ind, given_value = get_strategies(
        "test_cond_mvn_exception_sigma",
    )

    if n % 3 != 0 and n % 2 != 0:
        # `sigma` is negative definite matrix
        with pytest.raises(ValueError) as e:
            cond_mvn(mean, sigma, dependent_ind, given_ind, given_value)
        assert "sigma is not positive-definite" in str(e.value)
    elif n % 2 != 0:
        # `sigma` is not symmetric
        with pytest.raises(ValueError) as e:
            cond_mvn(mean, sigma, dependent_ind, given_ind, given_value)
        assert "sigma is not a symmetric matrix" in str(e.value)
    else:
        cond_mvn(mean, sigma, dependent_ind, given_ind, given_value)
