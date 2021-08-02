"""Tests for the Shapley effects.

This module contains tests for the Shapley effects. Tests are taken from:

- Iooss, B. and Prieur, C. (2019). Shapley effects for sensitivity analysis with
correlated inputs: comparisons with Sobol’ indices, numerical estimation and
applications, tech. report, Hyper articles en ligne.

- Plischke, E., Rabitti, G., & Borgonovo, E. (2020). Computing shapley effects for
sensitivity analysis. arXiv preprint.

"""
import itertools

import chaospy as cp
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal as aaae

from econsa.shapley import _r_condmvn
from econsa.shapley import get_permutations
from econsa.shapley import get_shapley


def test_get_permutations():
    method = "random"
    n_inputs = 4
    n_perms = 24
    seed = 123

    generated_permutations, generated_n_perms = get_permutations(method, n_inputs, n_perms, seed)

    all_permutations = np.asarray(list(itertools.permutations(range(n_inputs), n_inputs)))

    assert generated_n_perms == n_perms, "n_perms different."

    for current_permutation in all_permutations:
        assert np.apply_along_axis(
            np.array_equal,
            1,
            generated_permutations,
            current_permutation,
        ).any(), "Arrays of permutations differ."


def test_additive():
    def additive_model(x):
        return x[:, 0] + x[:, 1] * x[:, 2]

    def x_all(n):
        return cp.MvNormal(mean, cov).sample(n)

    def x_cond(n, subset_j, subsetj_conditional, xjc):
        if subsetj_conditional is None:
            cov_int = np.array(cov).take(subset_j, axis=1)[subset_j]
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

    var_1 = 1
    var_2 = 1
    var_3 = 1
    rho = 0.3

    covariance = rho * np.sqrt(var_1) * np.sqrt(var_3)

    var_y = var_1 + var_2 * var_3

    true_shapley_1 = (
        (var_1 * (1 - ((rho ** 2) / 2))) + (((var_2 * var_3) * (rho ** 2)) / 6)
    ) / var_y
    true_shapley_2 = (((var_2 * var_3) * (3 + (rho ** 2))) / 6) / var_y
    true_shapley_3 = (
        ((var_1 * (rho ** 2)) / 2) + (((var_2 * var_3) * (3 - (2 * (rho ** 2)))) / 6)
    ) / var_y

    n_inputs = 3
    mean = np.zeros(n_inputs)
    cov = np.array([[var_1, 0, covariance], [0, var_2, 0], [covariance, 0, var_3]])

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[
            [true_shapley_1, true_shapley_2, true_shapley_3],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        index=names,
        columns=col,
    ).T

    method = "exact"
    n_perms = None
    n_output = 10 ** 4
    n_outer = 3 * 10 ** 4
    n_inner = 3

    calculated = get_shapley(
        method,
        additive_model,
        x_all,
        x_cond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    assert_allclose(calculated["Shapley effects"], expected["Shapley effects"], rtol=0.02)


def test_ishigami():
    def ishigami_function(x):
        return np.sin(x[:, 0]) * (1 + 0.1 * (x[:, 2] ** 4)) + 7 * (np.sin(x[:, 1]) ** 2)

    def x_all(n):
        distribution = cp.Iid(cp.Uniform(lower, upper), n_inputs)
        return distribution.sample(n)

    def x_cond_uniform(n, subset_j, subsetj_conditional, xjc):
        distribution = cp.Iid(cp.Uniform(lower, upper), len(subset_j))
        return distribution.sample(n)

    n_inputs = 3
    lower = -np.pi
    upper = np.pi

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[
            [0.4358, 0.4424, 0.1218],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        index=names,
        columns=col,
    ).T

    method = "exact"
    n_perms = None
    n_output = 10 ** 4
    n_outer = 10 ** 4
    n_inner = 3

    np.random.seed(123)

    calculated = get_shapley(
        method,
        ishigami_function,
        x_all,
        x_cond_uniform,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    assert_allclose(calculated["Shapley effects"], expected["Shapley effects"], rtol=0.09)


def test_linear_three_inputs():
    def linear_model(x):
        beta = np.array([[beta_1], [beta_2], [beta_3]])
        return x.dot(beta)

    def x_all(n):
        return cp.MvNormal(mean, cov).sample(n)

    def x_cond(n, subset_j, subsetj_conditional, xjc):
        if subsetj_conditional is None:
            cov_int = np.array(cov).take(subset_j, axis=1)[subset_j]
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

    # Set parameters.
    beta_1 = 1.3
    beta_2 = 1.5
    beta_3 = 2.5
    var_1 = 16
    var_2 = 4
    var_3 = 9
    rho = 0.3

    covariance = rho * np.sqrt(var_2) * np.sqrt(var_3)

    component_1 = beta_1 ** 2 * var_1
    component_2 = beta_2 ** 2 * var_2
    component_3 = beta_3 ** 2 * var_3
    var_y = component_1 + component_2 + component_3 + 2 * covariance * beta_2 * beta_3
    share = 0.5 * (rho ** 2)

    true_shapley_1 = (component_1) / var_y
    true_shapley_2 = (
        component_2 + covariance * beta_2 * beta_3 + share * (component_3 - component_2)
    ) / var_y
    true_shapley_3 = (
        component_3 + covariance * beta_2 * beta_3 + share * (component_2 - component_3)
    ) / var_y

    n_inputs = 3
    mean = np.zeros(n_inputs)
    cov = np.array([[var_1, 0, 0], [0, var_2, covariance], [0, covariance, var_3]])

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[[true_shapley_1, true_shapley_2, true_shapley_3]],
        index=names,
        columns=col,
    ).T

    method = "exact"
    n_perms = None
    n_output = 10 ** 4
    n_outer = 10 ** 4
    n_inner = 3

    np.random.seed(1234)

    calculated = get_shapley(
        method,
        linear_model,
        x_all,
        x_cond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    assert_allclose(calculated["Shapley effects"], expected["Shapley effects"], rtol=0.02)


def test_linear_two_inputs():
    def linear_model(x):
        beta = np.array([[beta_1], [beta_2]])
        return x.dot(beta)

    def x_all(n):
        return cp.MvNormal(mean, cov).sample(n)

    def x_cond(n, subset_j, subsetj_conditional, xjc):
        if subsetj_conditional is None:
            cov_int = np.array(cov).take(subset_j, axis=1)[subset_j]
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

    # Set parameters.
    beta_1 = 1.3
    beta_2 = 1.5
    var_1 = 16
    var_2 = 4
    rho = 0.3

    covariance = rho * np.sqrt(var_1) * np.sqrt(var_2)

    component_1 = beta_1 ** 2 * var_1
    component_2 = beta_2 ** 2 * var_2
    var_y = component_1 + 2 * covariance * beta_1 * beta_2 + component_2
    share = 0.5 * (rho ** 2)

    true_shapley_1 = (
        component_1 * (1 - share) + covariance * beta_1 * beta_2 + component_2 * share
    ) / var_y
    true_shapley_2 = (
        component_2 * (1 - share) + covariance * beta_1 * beta_2 + component_1 * share
    ) / var_y

    n_inputs = 2
    mean = np.zeros(n_inputs)
    cov = np.array([[var_1, covariance], [covariance, var_2]])

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]
    names = ["Shapley effects", "std. errors", "CI_min", "CI_max"]

    expected = pd.DataFrame(
        data=[[true_shapley_1, true_shapley_2]],
        index=names,
        columns=col,
    ).T

    method = "exact"
    n_perms = None
    n_output = 10 ** 4
    n_outer = 10 ** 4
    n_inner = 3

    np.random.seed(1234)

    calculated = get_shapley(
        method,
        linear_model,
        x_all,
        x_cond,
        n_perms,
        n_inputs,
        n_output,
        n_outer,
        n_inner,
    )

    assert_allclose(calculated["Shapley effects"], expected["Shapley effects"], rtol=0.02)


def test_get_shapley_exact():
    def gaussian_model(x):
        # return np.sum(x)
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
        # return np.sum(x)

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
    n_perms = 6
    n_output = 10 ** 4
    n_outer = 10 ** 4
    n_inner = 10 ** 2
    seed = 1234

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
        seed,
    )

    assert_allclose(calculated, expected, rtol=0.03)
