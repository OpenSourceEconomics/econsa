"""Test for the DLR estimators of quantile based global sensitivity measures.

Analytical values of linear model with normally distributed variables
are used as benchmarks for verification of numerical estimates.
"""
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from quantile_measures_double_loop import dlr_mcs_quantile
from scipy.stats import norm
from temfpy.uncertainty_quantification import simple_linear_function


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

    # inverse error function
    phi_inv = norm.ppf(alp)

    # q_2: PDF of the out put Y(Eq.30)
    expect_q_2 = []

    for a in range(len(alp)):
        q_2_a = []
        for i in range(n_params_1):
            q_2_i = (
                cov_1[i, i]
                + phi_inv[a] ** 2
                * (
                    np.sqrt(np.trace(cov_1))
                    - np.sqrt(sum(cov_1[j, j] for j in range(n_params_1) if j != i))
                )
                ** 2
            )
            q_2_a.append(q_2_i)
        expect_q_2.append(q_2_a)

    expect_q_2 = np.vstack(expect_q_2).reshape((len(alp), n_params_1))

    # Q_2: normalized quantile based sensitivity measure 2.(Eq.14)
    expect_norm_q_2 = []

    for a in range(len(alp)):
        norm_q_2_a = []
        for i in range(n_params_1):
            norm_q_2_i = expect_q_2[a, i] / sum(expect_q_2[a])
            norm_q_2_a.append(norm_q_2_i)
        expect_norm_q_2.append(norm_q_2_a)

    expect_norm_q_2 = np.hstack(expect_norm_q_2).reshape((len(alp), n_params_1))

    quantile_measures_expected = expect_norm_q_2

    out = {
        "func": simple_linear_function_transposed,
        "n_params": n_params_1,
        "loc": mean_1,
        "scale": cov_1,
        "dist_type": "Normal",
        "n_draws": 2 ** 13,
        "quantile_measures_expected": quantile_measures_expected,
    }

    return out


def test_quantile_measures_first_example(first_example_fixture):
    quantile_measures_expected = first_example_fixture["quantile_measures_expected"]
    func = first_example_fixture["func"]
    n_params = first_example_fixture["n_params"]
    loc = first_example_fixture["loc"]
    scale = first_example_fixture["scale"]
    dist_type = first_example_fixture["dist_type"]
    n_draws = first_example_fixture["n_draws"]

    quantile_measures = dlr_mcs_quantile(
        func=func,
        n_params=n_params,
        loc=loc,
        scale=scale,
        dist_type=dist_type,
        n_draws=n_draws,
    )

    assert_almost_equal(quantile_measures[3], quantile_measures_expected, decimal=2)
