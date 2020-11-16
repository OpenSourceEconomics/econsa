"""Test for quantile based global sensitivity measures.

Analytical values of linear model with normally distributed variables
are calculated as benchmarks for verification of numerical estimates.
"""
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.stats import norm
from temfpy.uncertainty_quantification import simple_linear_function

from econsa.quantile_measures import mc_quantile_measures


@pytest.fixture
def first_example_fixture():
    """First example test case."""

    def simple_linear_function_transposed(x):
        """Simple linear function model but with variables stored in columns."""
        return simple_linear_function(x.T)

    mean_1 = np.array([1, 3, 5, 7])
    cov_1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 2.25, 0, 0],
            [0, 0, 4, 0],
            [0, 0, 0, 6.25],
        ],
    )
    n_params_1 = len(mean_1)

    # range of alpha
    dalp = (0.98 - 0.02) / 30
    alp = np.arange(0.02, 0.98 + dalp, dalp)  # len(alp) = 31

    # q_2: PDF of the out put Y(Eq.30)
    q_2_true = [
        (
            cov_1[i, i]
            + norm.ppf(a) ** 2
            * (
                np.sqrt(np.trace(cov_1))
                - np.sqrt(sum(cov_1[j, j] for j in range(n_params_1) if j != i))
            )
            ** 2
        )
        for a in alp
        for i in range(n_params_1)
    ]
    # reshape
    q_2_true = np.vstack(q_2_true).reshape((len(alp), n_params_1))

    # Q_2: normalized quantile based sensitivity measure 2.(Eq.14)
    norm_q_2_true = [q[i] / sum(q) for q in q_2_true for i in range(n_params_1)]
    norm_q_2_true = np.hstack(norm_q_2_true).reshape((len(alp), n_params_1))

    quantile_measures_true = norm_q_2_true

    out = {
        "func": simple_linear_function_transposed,
        "n_params": n_params_1,
        "loc": mean_1,
        "scale": cov_1,
        "dist_type": "Normal",
        "n_draws_dlr": 2 ** 13,
        "n_draws_bf": 3000,
        "quantile_measures_true": quantile_measures_true,
    }

    return out


def test_quantile_measures_first_example(first_example_fixture):
    quantile_measures_true = first_example_fixture["quantile_measures_true"]
    func = first_example_fixture["func"]
    n_params = first_example_fixture["n_params"]
    loc = first_example_fixture["loc"]
    scale = first_example_fixture["scale"]
    dist_type = first_example_fixture["dist_type"]
    n_draws_dlr = first_example_fixture["n_draws_dlr"]
    n_draws_br = first_example_fixture["n_draws_bf"]

    for estimator, n_draws in zip(
        ["DLR", "brute force"],
        [n_draws_dlr, n_draws_br],
    ):
        quantile_measures_solve = mc_quantile_measures(
            estimator,
            func=func,
            n_params=n_params,
            loc=loc,
            scale=scale,
            dist_type=dist_type,
            n_draws=n_draws,
        )
        assert_array_almost_equal(
            quantile_measures_solve.loc["Q_2"],
            quantile_measures_true,
            decimal=2,
        )
