"""Tests for quantile based sensitivity measures.

We implement tests replicating the results presented in Kucherenko et al. 2019.
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

    # Analytical values of linear model with normally distributed variables
    # are used as benchmarks for the first test case.
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

    # Set numerical estimates of brute force estimator with 2^13 draws as benchmark
    norm_q_2_expected = np.array(
        [
            [0.181, 0.324, 0.177, 0.318],
            [0.213, 0.288, 0.217, 0.282],
            [0.228, 0.273, 0.226, 0.272],
            [0.231, 0.267, 0.234, 0.268],
            [0.243, 0.258, 0.242, 0.257],
            [0.247, 0.257, 0.244, 0.253],
            [0.25, 0.252, 0.249, 0.249],
            [0.252, 0.249, 0.251, 0.248],
            [0.253, 0.248, 0.252, 0.246],
            [0.253, 0.248, 0.252, 0.246],
            [0.254, 0.248, 0.253, 0.246],
            [0.253, 0.248, 0.252, 0.246],
            [0.251, 0.249, 0.253, 0.247],
            [0.25, 0.25, 0.251, 0.249],
            [0.248, 0.253, 0.249, 0.25],
            [0.247, 0.253, 0.249, 0.251],
            [0.247, 0.253, 0.248, 0.252],
            [0.247, 0.254, 0.247, 0.252],
            [0.247, 0.254, 0.247, 0.252],
            [0.245, 0.256, 0.246, 0.253],
            [0.243, 0.257, 0.243, 0.256],
            [0.244, 0.257, 0.244, 0.254],
            [0.246, 0.255, 0.246, 0.253],
            [0.246, 0.254, 0.246, 0.253],
            [0.248, 0.252, 0.248, 0.251],
            [0.253, 0.249, 0.251, 0.247],
            [0.256, 0.244, 0.257, 0.244],
            [0.269, 0.234, 0.262, 0.235],
            [0.272, 0.228, 0.278, 0.222],
            [0.275, 0.221, 0.285, 0.219],
            [0.303, 0.203, 0.296, 0.198],
        ],
    )

    n_params = norm_q_2_expected.shape[1]

    # Here we test the performance of DLR approach with 2^10 draws, which only achieves
    # 1 decimal digit precision when compared with the brute force estimates with 3000 draws.
    # To achieve good convergence the DLR draw should be lager.
    for estimator, n_draws in zip(
        ["DLR", "brute force"],
        [2 ** 10, 3000],
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
            norm_q_2_expected,
            decimal=1,
        )


def test_3():
    """Third test case."""

    # Objective function
    def ishigami_transposed(x, a=7, b=0.1):
        """Ishigami function with variables stored in columns."""
        return ishigami(x.T, a=7, b=0.1)

    # benchmark: mean values of brute force estimates with 3000 draws
    # and DLR estimates with 2^14 draws.
    norm_q_2_expected = np.array(
        [
            [0.511, 0.204, 0.285],
            [0.406, 0.35, 0.244],
            [0.313, 0.45, 0.237],
            [0.255, 0.545, 0.199],
            [0.225, 0.609, 0.166],
            [0.214, 0.652, 0.134],
            [0.22, 0.667, 0.112],
            [0.238, 0.663, 0.098],
            [0.268, 0.645, 0.087],
            [0.304, 0.625, 0.071],
            [0.341, 0.605, 0.054],
            [0.368, 0.59, 0.042],
            [0.393, 0.581, 0.026],
            [0.411, 0.578, 0.011],
            [0.423, 0.574, 0.003],
            [0.425, 0.575, 0.0],
            [0.422, 0.575, 0.003],
            [0.41, 0.579, 0.011],
            [0.393, 0.581, 0.026],
            [0.368, 0.591, 0.041],
            [0.341, 0.606, 0.053],
            [0.308, 0.62, 0.072],
            [0.271, 0.641, 0.088],
            [0.242, 0.657, 0.101],
            [0.223, 0.66, 0.117],
            [0.219, 0.64, 0.14],
            [0.226, 0.603, 0.171],
            [0.261, 0.537, 0.203],
            [0.317, 0.443, 0.24],
            [0.408, 0.347, 0.244],
            [0.504, 0.205, 0.291],
        ],
    )

    n_params = norm_q_2_expected.shape[1]
    # lower bound of uniform distribution
    lower_bound = -np.pi
    # interval of uniform distribution
    interval = 2 * np.pi

    # Here we test naive monte carlo estimates by specifying `sampling_scheme` to "random".
    for estimator, n_draws in zip(
        ["DLR", "brute force"],
        [2 ** 14, 3000],
    ):
        norm_q_2_solve = mc_quantile_measures(
            estimator=estimator,
            func=ishigami_transposed,
            n_params=n_params,
            loc=lower_bound,
            scale=interval,
            dist_type="Uniform",
            n_draws=n_draws,
            sampling_scheme="random",
        )
        assert_array_almost_equal(
            norm_q_2_solve.loc["Q_2"],
            norm_q_2_expected,
            decimal=2,
        )
