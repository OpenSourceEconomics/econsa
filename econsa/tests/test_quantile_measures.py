"""This module contains tests for quantile based sensitivity measures.

We implement tests replicating the results presented in Kucherenko et al. 2019.
Analytical values of linear model with normally distributed variables
are used as benchmarks for the first test case. Numerical
estimates of double loop reordering approach with 2^13 draws are
used as benchmarks for verifying other cases.
"""
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.stats import norm
from temfpy.uncertainty_quantification import ishigami
from temfpy.uncertainty_quantification import simple_linear_function

from econsa.quantile_measures import mc_quantile_measures


def test_1():
    """First test case."""

    # Objective function
    def simple_linear_function_transposed(x):
        """Simple linear function model but with variables stored in columns."""
        return simple_linear_function(x.T)

    mean = np.array([1, 3, 5, 7])
    cov = np.array(
        [
            [1, 0, 0, 0],
            [0, 2.25, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 6.25],
        ],
    )
    n_params = len(mean)

    # range of alpha
    dalp = (0.98 - 0.02) / 30
    alp = np.arange(0.02, 0.98 + dalp, dalp)  # len(alp) = 31

    # q_2: PDF of the out put Y(Eq.30)
    q_2_true = [
        (
            cov[i, i]
            + norm.ppf(a) ** 2
            * (np.sqrt(np.trace(cov)) - np.sqrt(sum(cov[j, j] for j in range(n_params) if j != i)))
            ** 2
        )
        for a in alp
        for i in range(n_params)
    ]
    # reshape
    q_2_true = np.vstack(q_2_true).reshape((len(alp), n_params))

    # Q_2: normalized quantile based sensitivity measure 2.(Eq.14)
    norm_q_2_true = [q[i] / sum(q) for q in q_2_true for i in range(n_params)]
    norm_q_2_true = np.hstack(norm_q_2_true).reshape((len(alp), n_params))

    for estimator, n_draws in zip(
        ["DLR", "brute force"],
        [2 ** 13, 3000],
    ):
        norm_q_2_solve = mc_quantile_measures(
            estimator=estimator,
            func=simple_linear_function_transposed,
            n_params=n_params,
            loc=mean,
            scale=cov,
            dist_type="Normal",
            n_draws=n_draws,
        )
        # Numerical approximation can be more precise with the increase of n_draws.
        assert_array_almost_equal(
            norm_q_2_solve.loc["Q_2"],
            norm_q_2_true,
            decimal=2,
        )


def test_2():
    """Second test case."""

    # Objective function
    def simple_linear_function_modified(x):
        """Simple linear function model with variables stored in columns and signs changed."""
        a = [1, -1, 1, -1]
        b = [q for q in x.T]
        return simple_linear_function([i * q for i, q in zip(a, b)])

    # Numerical estimates using double loop reordering approach with 2^13 draws
    norm_q_2_true = np.array(
        [
            [0.178, 0.344, 0.17, 0.309],
            [0.2, 0.305, 0.2, 0.295],
            [0.217, 0.287, 0.214, 0.281],
            [0.226, 0.273, 0.226, 0.275],
            [0.236, 0.265, 0.234, 0.264],
            [0.243, 0.259, 0.241, 0.257],
            [0.248, 0.255, 0.246, 0.252],
            [0.25, 0.25, 0.248, 0.251],
            [0.254, 0.248, 0.251, 0.247],
            [0.255, 0.246, 0.252, 0.246],
            [0.255, 0.245, 0.253, 0.246],
            [0.255, 0.246, 0.252, 0.247],
            [0.255, 0.246, 0.252, 0.248],
            [0.253, 0.247, 0.25, 0.25],
            [0.252, 0.248, 0.249, 0.25],
            [0.252, 0.248, 0.249, 0.251],
            [0.251, 0.249, 0.248, 0.252],
            [0.25, 0.251, 0.247, 0.253],
            [0.249, 0.251, 0.246, 0.254],
            [0.248, 0.253, 0.245, 0.254],
            [0.248, 0.253, 0.244, 0.255],
            [0.249, 0.251, 0.245, 0.254],
            [0.25, 0.25, 0.247, 0.253],
            [0.253, 0.247, 0.249, 0.251],
            [0.256, 0.244, 0.251, 0.249],
            [0.263, 0.24, 0.258, 0.239],
            [0.268, 0.233, 0.264, 0.235],
            [0.28, 0.223, 0.271, 0.226],
            [0.286, 0.213, 0.285, 0.216],
            [0.3, 0.197, 0.3, 0.203],
            [0.318, 0.171, 0.335, 0.176],
        ],
    )

    n_params = norm_q_2_true.shape[1]

    for estimator, n_draws in zip(
        ["DLR", "brute force"],
        [2 ** 13, 3000],
    ):
        norm_q_2_solve = mc_quantile_measures(
            estimator=estimator,
            func=simple_linear_function_modified,
            n_params=n_params,
            loc=0,
            scale=1,
            dist_type="Exponential",
            n_draws=n_draws,
        )
        assert_array_almost_equal(
            norm_q_2_solve.loc["Q_2"],
            norm_q_2_true,
            decimal=1,
        )


def test_3():
    """Third test case."""

    # Objective function
    def ishigami_transposed(x, a=7, b=0.1):
        """Ishigami function with variables stored in columns."""
        return ishigami(x.T, a=7, b=0.1)

    # Numerical estimates using double loop reordering approach with 2^13 draws
    norm_q_2_true = np.array(
        [
            [0.513, 0.2, 0.287],
            [0.419, 0.356, 0.226],
            [0.323, 0.459, 0.218],
            [0.264, 0.555, 0.182],
            [0.234, 0.621, 0.145],
            [0.22, 0.658, 0.122],
            [0.226, 0.676, 0.098],
            [0.245, 0.672, 0.083],
            [0.275, 0.656, 0.069],
            [0.309, 0.63, 0.061],
            [0.346, 0.608, 0.046],
            [0.373, 0.591, 0.035],
            [0.399, 0.581, 0.02],
            [0.416, 0.574, 0.01],
            [0.428, 0.569, 0.003],
            [0.431, 0.569, 0.0],
            [0.429, 0.569, 0.003],
            [0.416, 0.573, 0.01],
            [0.401, 0.579, 0.02],
            [0.377, 0.588, 0.035],
            [0.349, 0.604, 0.047],
            [0.318, 0.62, 0.062],
            [0.28, 0.649, 0.071],
            [0.25, 0.665, 0.085],
            [0.231, 0.669, 0.1],
            [0.227, 0.648, 0.125],
            [0.236, 0.617, 0.148],
            [0.271, 0.546, 0.183],
            [0.33, 0.452, 0.218],
            [0.419, 0.351, 0.23],
            [0.514, 0.199, 0.288],
        ],
    )

    n_params = norm_q_2_true.shape[1]
    # lower bound of uniform distribution
    lower_bound = -np.pi
    # interval of uniform distribution
    interval = 2 * np.pi

    for estimator, n_draws in zip(
        ["DLR", "brute force"],
        [2 ** 13, 3000],
    ):
        norm_q_2_solve = mc_quantile_measures(
            estimator=estimator,
            func=ishigami_transposed,
            n_params=n_params,
            loc=lower_bound,
            scale=interval,
            dist_type="Uniform",
            n_draws=n_draws,
        )
        assert_array_almost_equal(
            norm_q_2_solve.loc["Q_2"],
            norm_q_2_true,
            decimal=1,
        )
