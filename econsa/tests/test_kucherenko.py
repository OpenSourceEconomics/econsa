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

import econsa.kucherenko
from econsa.aleeciu import build_cov_mu
from econsa.aleeciu import kucherenko_indices


@pytest.fixture
def first_example_fixture():
    """First example test case. Results are given in [Table 1].
    """

    def func1(args):
        """Test function from Kucherenko et al. 2012."""
        result = np.sum(args, axis=1)
        return result

    problem = {
        "num_vars": 3,
        "dist": np.array(3 * ["norm"]),
        "prms": np.array([[0.0, 1], [0, 1], [0, 4]]),
    }

    def create_covariance_and_mean(rho, factor):
        df_cov = pd.DataFrame([[1.0, 0, 0], [0, 1, rho * 2], [0, rho * 2, 4]])
        cov, mean = build_cov_mu(df_cov, np.zeros((3, 1)), [factor])
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
    df_expected.index = pd.MultiIndex.from_tuples(
        list_of_tuples, names=["rho", "variable"]
    )

    out = {
        "func": func1,
        "create_covariance_and_mean": create_covariance_and_mean,
        "problem": problem,
        "N": 5000,
        "df_expected": df_expected,
    }

    return out


def test_kucherenko_indices_first_example(first_example_fixture):
    df_expected = first_example_fixture["df_expected"]
    func = first_example_fixture["func"]
    problem = first_example_fixture["problem"]
    N = first_example_fixture["N"]

    for rho, factor in df_expected.index:
        cov, mean = first_example_fixture["create_covariance_and_mean"](rho, factor)

        first_order, total = kucherenko_indices(func, cov, mean, problem, N)

        assert first_order == pytest.approx(
            df_expected.loc[(rho, factor), "first_order"], abs=0.01
        )
        assert total == pytest.approx(df_expected.loc[(rho, factor), "total"], abs=0.01)


@pytest.fixture
def second_example_fixture():
    """Second example test case. Results are given in [Table 2].
    """

    def func2(args):
        """Test function from Kucherenko et al. 2012."""
        a, b, c, d = np.hsplit(args, 4)
        result = a * c + b * d
        return result

    problem = {
        "num_vars": 4,
        "dist": np.array(4 * ["norm"]),
        "prms": np.array([[0.0, 16], [0, 4], [250, 40_000], [400, 90_000]]),
    }

    def create_covariance_and_mean(factor):
        df_cov = pd.DataFrame(
            [
                [16.0, 2.4, 0, 0],
                [2.4, 4, 0, 0],
                [0, 0, 40_000, -18_000],
                [0, 0, -18_000, 90_000],
            ]
        )
        cov, mean = build_cov_mu(
            df_cov, np.array([0.0, 0, 250, 400]).reshape(4, 1), [factor]
        )
        return cov, mean

    df_expected = pd.DataFrame(
        [[0.507, 0.492], [0.399, 0.300], [0.000, 0.192], [0.000, 0.108]],
        columns=["first_order", "total"],
    )

    out = {
        "func": func2,
        "create_covariance_and_mean": create_covariance_and_mean,
        "problem": problem,
        "N": 25000,
        "df_expected": df_expected,
    }
    return out


def test_kucherenko_indices_second_example(second_example_fixture):
    df_expected = second_example_fixture["df_expected"]
    func = second_example_fixture["func"]
    problem = second_example_fixture["problem"]
    N = second_example_fixture["N"]

    for i in df_expected.index:
        cov, mean = second_example_fixture["create_covariance_and_mean"](factor=i)

        first_order, total = kucherenko_indices(func, cov, mean, problem, N)

        assert first_order == pytest.approx(df_expected.loc[i, "first_order"], abs=0.01)
        assert total == pytest.approx(df_expected.loc[i, "total"], abs=0.01)


@pytest.fixture
def first_example_fixture_my():
    """First example test case. Results are given in [Table 1].
    """

    def func1(args):
        """Test function from Kucherenko et al. 2012."""
        result = np.sum(args, axis=1)
        return result

    def create_covariance_and_mean(rho, factor):
        df_cov = pd.DataFrame([[1.0, 0, 0], [0, 1, rho * 2], [0, rho * 2, 4]])
        cov, mean = df_cov.values, np.zeros(3)
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
    df_expected.index = pd.MultiIndex.from_tuples(
        list_of_tuples, names=["rho", "variable"]
    )

    out = {
        "func": func1,
        "create_covariance_and_mean": create_covariance_and_mean,
        "n_draws": 10_000,
        "df_expected": df_expected,
    }

    return out


def test_kucherenko_indices_first_example_my(first_example_fixture_my):
    df_expected = first_example_fixture_my["df_expected"]
    func = first_example_fixture_my["func"]
    n_draws = first_example_fixture_my["n_draws"]

    for rho in df_expected.index.get_level_values("rho").unique():
        cov, mean = first_example_fixture_my["create_covariance_and_mean"](rho, 1)

        df_indices = econsa.kucherenko.kucherenko_indices(
            func=func, sampling_mean=mean, sampling_cov=cov, n_draws=n_draws
        )

        for var, typ in df_indices.index:
            assert df_indices.loc[(var, typ), "value"] == pytest.approx(
                df_expected.loc[(rho, var), typ], abs=0.01
            )
