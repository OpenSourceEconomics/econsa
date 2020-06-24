"""Tests for the sampling.

This module contains all tests for the sampling setup.

"""
import numpy as np
import pytest
import rpy2.robjects.packages as rpackages
from numpy.random import RandomState
from rpy2 import robjects
from rpy2.robjects import numpy2ri

from econsa.sampling import condMVN

# Since NumPy v1.17: from numpy.random import default_rng


# Import R modules
base = rpackages.importr("base")
stats = rpackages.importr("stats")
utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)
utils.install_packages("condMVNorm")
condMVNorm = rpackages.importr("condMVNorm")

# Import numpy.random generator
# Since NumPy v1.17: rng = default_rng()
rs = RandomState()


def test_sampling():
    pass


def get_strategies(name):
    if name == "condMVN":
        n = rs.randint(low=4, high=20)
        mean = rs.randint(low=-2, high=2, size=n)
        sigma = rs.standard_normal(size=(n, n))
        sigma = sigma @ sigma.T
        dependent_n = rs.randint(low=1, high=n - 2)
        dependent = rs.choice(range(0, n), replace=False, size=dependent_n)
        given_ind = [x for x in range(0, n) if x not in dependent]
        x_given = rs.randint(low=-2, high=2, size=len(given_ind))
        strategy = (n, mean, sigma, dependent, given_ind, x_given)
    elif name == "condMVN_exception":
        n = rs.randint(low=4, high=20)
        mean = rs.randint(low=-2, high=2, size=n)
        sigma = rs.standard_normal(size=(n, n))
        sigma = sigma @ sigma.T
        dependent_n = rs.randint(low=1, high=n - 2)
        dependent = rs.choice(range(0, n), replace=False, size=dependent_n)
        given_ind = [x for x in range(0, n) if x not in dependent] if n % 2 == 0 else []
        x_given = (
            rs.randint(low=-2, high=2, size=len(given_ind) + 1) if n % 3 == 0 else []
        )
        strategy = (n, mean, sigma, dependent, given_ind, x_given)
    else:
        raise NotImplementedError
    return strategy


def test_condMVN():
    """Test condMVN against the original from R package condMVNorm.
    """
    # Evaluate Python code
    n, mean, sigma, dependent_ind, given_ind, x_given = get_strategies("condMVN")
    cond_mean, cond_var = condMVN(mean, sigma, dependent_ind, given_ind, x_given)

    # Evaluate R code
    numpy2ri.activate()
    mean_r = robjects.FloatVector(mean)
    sigma_r = base.matrix(sigma, n, n)
    dependent_ind_r = robjects.IntVector([x + 1 for x in dependent_ind])
    given_ind_r = robjects.IntVector([x + 1 for x in given_ind])
    x_given_r = robjects.IntVector(x_given)
    condMean_r, condVar_r = condMVNorm.condMVN(
        mean=mean_r,
        sigma=sigma_r,
        dependent=dependent_ind_r,
        given=given_ind_r,
        X=x_given_r,
    )
    numpy2ri.deactivate()

    condMean_r = np.array(condMean_r)
    condVar_r = np.array(condVar_r)

    # Comparison
    np.testing.assert_allclose(cond_mean, condMean_r)
    np.testing.assert_allclose(cond_var, condVar_r)


def test_condMVN_exception():
    """Test condMVN raises exceptions when invalid variables are passed.
    """
    n, mean, sigma, dependent_ind, given_ind, x_given = get_strategies(
        "condMVN_exception",
    )
    if n % 2 == 0 and n % 3 == 0:
        with pytest.raises(ValueError):
            condMVN(mean, sigma, dependent_ind, given_ind, x_given)
    elif n % 2 == 0:
        with pytest.raises(ValueError):
            condMVN(mean, sigma, dependent_ind, given_ind, x_given)
    elif n % 3 == 0:
        with pytest.raises(TypeError):
            condMVN(mean, sigma, dependent_ind, given_ind, x_given)
    else:
        condMVN(mean, sigma, dependent_ind, given_ind, x_given)
