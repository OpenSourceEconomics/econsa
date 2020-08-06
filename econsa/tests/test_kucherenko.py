"""Tests for the generalized sobol indices first described in Kucherenko et al. 2012.

We test that the implemented functions replicate the results presented in Kucherenko
et al. 2012.

All references to Tables, Equation and so on correspond to references in Kucherenko et
al. 2012.
"""
from itertools import product

import numpy as np
import pandas as pd
import pytest

from econsa.kucherenko import kucherenko_indices


@pytest.fixture
def first_example_fixture():
    """First example test case. Results are given in [Table 1].
    """

    def func1(args):
        """Test function from Kucherenko et al. 2012."""
        result = np.sum(args, axis=1)
        return result

    def create_covariance_and_mean(rho):
        cov = np.array([[1.0, 0, 0], [0, 1, rho * 2], [0, rho * 2, 4]])
        mean = np.zeros(3)
        return cov, mean

    # Analytical first order and total indices in [Table 1].
    df_expected = pd.DataFrame(
        [
            [0.167, 0.167],
            [0.167, 0.167],
            [0.667, 0.667],
            [0.125, 0.125],
            [0.500, 0.094],
            [0.781, 0.375],
            [0.250, 0.250],
            [0.000, 0.188],
            [0.563, 0.750],
            [0.109, 0.109],
            [0.735, 0.039],
            [0.852, 0.157],
            [0.357, 0.357],
            [0.129, 0.129],
            [0.514, 0.514],
        ],
        columns=["first_order", "total"],
    )
    list_of_tuples = list(product([0.0, 0.5, -0.5, 0.8, -0.8], [0, 1, 2]))
    df_expected.index = pd.MultiIndex.from_tuples(list_of_tuples, names=["rho", "var"])

    out = {
        "func": func1,
        "create_covariance_and_mean": create_covariance_and_mean,
        "n_draws": 10_000,
        "df_expected": df_expected,
    }

    return out


def test_kucherenko_indices_first_example(first_example_fixture):
    df_expected = first_example_fixture["df_expected"]
    func = first_example_fixture["func"]
    n_draws = first_example_fixture["n_draws"]

    for rho in df_expected.index.get_level_values("rho").unique():
        cov, mean = first_example_fixture["create_covariance_and_mean"](rho)

        df_indices = kucherenko_indices(
            func=func, sampling_mean=mean, sampling_cov=cov, n_draws=n_draws,
        )

        for var, typ in df_indices.index:
            assert df_indices.loc[(var, typ), "value"] == pytest.approx(
                df_expected.loc[(rho, var), typ], abs=0.01,
            )


@pytest.fixture
def second_example_fixture():
    """Second example test case. Results are given in [Table 2].
    """

    def func2(args):
        """Test function from Kucherenko et al. 2012."""
        a, b, c, d = np.hsplit(args, 4)
        result = a * c + b * d
        return result

    sampling_mean = np.array([0.0, 0, 250, 400])
    sampling_cov = np.array(
        [
            [16.0, 2.4, 0, 0],
            [2.4, 4, 0, 0],
            [0, 0, 40_000, -18_000],
            [0, 0, -18_000, 90_000],
        ],
    )

    df_expected = pd.DataFrame(
        [[0.507, 0.492], [0.399, 0.300], [0.000, 0.192], [0.000, 0.108]],
        columns=["first_order", "total"],
    )

    out = {
        "func": func2,
        "sampling_mean": sampling_mean,
        "sampling_cov": sampling_cov,
        "n_draws": 25_000,
        "df_expected": df_expected,
    }
    return out


def test_kucherenko_indices_second_example_my(second_example_fixture):
    df_expected = second_example_fixture["df_expected"]
    func = second_example_fixture["func"]
    n_draws = second_example_fixture["n_draws"]
    sample_mean = second_example_fixture["sampling_mean"]
    sample_cov = second_example_fixture["sampling_cov"]

    df_indices = kucherenko_indices(
        func, sampling_mean=sample_mean, sampling_cov=sample_cov, n_draws=n_draws,
    )

    for var, typ in df_indices.index:
        assert df_indices.loc[(var, typ), "value"] == pytest.approx(
            df_expected.loc[var, typ], abs=0.01,
        )
