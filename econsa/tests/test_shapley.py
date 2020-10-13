"""Tests for the Shapley effects.

This module contains all tests for th Shapley effects.

"""
import chaospy as cp
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal as aaae

from econsa.shapley import _r_condmvn
from econsa.shapley import get_shapley


def test_get_shapley_exact():
    def gaussian_model(x):
        return np.sum(x, 1)

    def x_all(n):
        distribution = cp.MvNormal(mean, cov)
        return distribution.sample(n)

    def x_cond(n, subset_j, subsetj_conditional, xjc):
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
                x_given=xjc,
            )

    np.random.seed(123)
    n_inputs = 3
    mean = np.zeros(3)
    cov = np.array([[1.0, 0, 0], [0, 1.0, 1.8], [0, 1.8, 4.0]])
    method = "exact"
    n_perms = None
    n_output = 10 ** 4
    n_outer = 10 ** 3
    n_inner = 10 ** 2

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[
            [0.101309, 0.418989, 0.479701],
            [0.00241549, 0.16297, 0.163071],
            [0.096575, 0.0995681, 0.160083],
            [0.106044, 0.73841, 0.79932],
        ],
        index=names,
        columns=col,
    ).T

    calculated = get_shapley(
        method,
        gaussian_model,
        x_all,
        x_cond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    aaae(calculated, expected)


def test_get_shapley_random():
    def gaussian_model(x):
        return np.sum(x, 1)

    def x_all(n):
        distribution = cp.MvNormal(mean, cov)
        return distribution.sample(n)

    def x_cond(n, subset_j, subsetj_conditional, xjc):
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
                x_given=xjc,
            )

    np.random.seed(123)
    n_inputs = 3
    mean = np.zeros(3)
    cov = np.array([[1.0, 0, 0], [0, 1.0, 1.8], [0, 1.8, 4.0]])
    method = "random"
    n_perms = 30000
    n_output = 10 ** 4
    n_outer = 1
    n_inner = 3

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[
            [0.107543, 0.414763, 0.477694],
            [0.00307984, 0.0032332, 0.0031896],
            [0.101507, 0.408426, 0.471442],
            [0.11358, 0.4211, 0.483945],
        ],
        index=names,
        columns=col,
    ).T

    calculated = get_shapley(
        method,
        gaussian_model,
        x_all,
        x_cond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    aaae(calculated, expected)
