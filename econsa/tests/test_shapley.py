"""Tests for the Shapley effects.

This module contains all tests for th Shapley effects.

"""

import numpy as np
import chaospy as cp
from numpy.testing import assert_array_almost_equal as aaae
from pandas.testing import assert_series_equal

from shapley import (
    _r_condmvn,
    get_shapley,
)


def test_get_shapley_exact():
    def gaussian_model(X):
        return np.sum(X, 1)

    def Xall(n):
        distribution = cp.MvNormal(mean, cov)
        return distribution.sample(n)

    def Xcond(n, subset_j, subsetj_conditional, xjc):
        if subsetj_conditional is None:
            cov_int = np.array(cov)
            cov_int = cov_int.take(subset_j, axis=1)
            cov_int = cov_int[subset_j]
            distribution = cp.MvNormal(mean[subset_j], cov_int)
            return distribution.sample(n)
        else:
            return _r_condmvn(
                n,
                mean=mean,
                cov=cov,
                dependent_ind=subset_j,
                given_ind=subsetj_conditional,
                X_given=xjc,
            )

    n_inputs = 3
    mean = np.zeros(3)
    cov = np.array([[1.0, 0, 0], [0, 1.0, 1.8], [0, 1.8, 4.0]])
    method = "exact"
    n_perms = None
    n_output = 10 ** 4
    n_outer = 10 ** 3
    n_inner = 10 ** 2

    get_shapley(
        method,
        gaussian_model,
        Xall,
        Xcond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )


def test_get_shapley_random():
    def gaussian_model(X):
        return np.sum(X, 1)

    def Xall(n):
        distribution = cp.MvNormal(mean, cov)
        return distribution.sample(n)

    def Xcond(n, subset_j, subsetj_conditional, xjc):
        if subsetj_conditional is None:
            cov_int = np.array(cov)
            cov_int = cov_int.take(subset_j, axis=1)
            cov_int = cov_int[subset_j]
            distribution = cp.MvNormal(mean[subset_j], cov_int)
            return distribution.sample(n)
        else:
            return _r_condmvn(
                n,
                mean=mean,
                cov=cov,
                dependent_ind=subset_j,
                given_ind=subsetj_conditional,
                X_given=xjc,
            )

    n_inputs = 3
    mean = np.zeros(3)
    cov = np.array([[1.0, 0, 0], [0, 1.0, 1.8], [0, 1.8, 4.0]])
    method = "random"
    n_perms = 6000
    n_output = 10 ** 4
    n_outer = 1
    n_inner = 3

    get_shapley(
        method,
        gaussian_model,
        Xall,
        Xcond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )
