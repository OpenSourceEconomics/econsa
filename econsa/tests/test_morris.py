import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal as aaae

from econsa.morris import _shift_cov
from econsa.morris import _shift_sample
from econsa.morris import _uniform_to_standard_normal
from econsa.morris import elementary_effects


def test_uniform_to_standard_normal():
    uni = np.array([0.02275013, 0.15865525, 0.5, 0.84134475, 0.97724987])
    expected = np.array([-2.0, -1, 0, 1, 2])
    calculated = _uniform_to_standard_normal(uni)
    aaae(calculated, expected)


def test_shift_sample_1d():
    calculated = _shift_sample(np.arange(5), 3)
    expected = np.array([3, 4, 0, 1, 2])
    aaae(calculated, expected)


def test_shift_sample_2d():
    calculated = _shift_sample(np.arange(10, dtype=float).reshape(2, -1), 3)
    expected = np.array([[3.0, 4.0, 0.0, 1.0, 2.0], [8.0, 9.0, 5.0, 6.0, 7.0]])
    aaae(calculated, expected)


def test_shift_sample_3d():
    sample = np.arange(20, dtype=float).reshape(2, 2, -1)
    k_arr = np.array([1, 3])
    calculated = _shift_sample(sample, k_arr.reshape(1, 2))
    expected = np.array(
        [
            [[1.0, 2.0, 3, 4, 0], [8.0, 9.0, 5.0, 6.0, 7.0]],
            [[11.0, 12.0, 13.0, 14.0, 10.0], [18.0, 19.0, 15.0, 16.0, 17.0]],
        ],
    )
    aaae(calculated, expected)


def test_shift_cov():
    np.random.seed(1234)
    true_cov = np.array(
        [
            [1, 0.1, 0.2, 0.3],
            [0.1, 2, 0.4, 0.5],
            [0.2, 0.4, 3, 0.6],
            [0.3, 0.5, 0.6, 4],
        ],
    )
    data = np.random.multivariate_normal(mean=np.zeros(4), cov=true_cov, size=20)
    df = pd.DataFrame(data=data)
    cov = df.cov().to_numpy()
    df2 = df[[2, 3, 0, 1]]
    expected = df2.cov().to_numpy()
    calculated = _shift_cov(cov, 2)
    aaae(calculated, expected)


def model_func(params):
    return 5


def test_ee():

    names = ["x1", "x2", "x3"]
    params = pd.DataFrame(columns=["value"], data=[0, 0.0, 0], index=names)

    cov = pd.DataFrame(
        data=[[1, 0.9, 0.4], [0.9, 1, 0.01], [0.4, 0.01, 1]],
        columns=names,
        index=names,
    )
    n_draws = 100

    elementary_effects(model_func, params, cov, n_draws)


def simple_linear_model(x):
    """For test case 1."""
    return np.sum(x)


def test_sampling_scheme():
    n_inputs = 3
    names = ["x1", "x2", "x3"]
    params = pd.DataFrame(columns=["value"], data=np.zeros(n_inputs), index=names)
    cov = pd.DataFrame(
        data=[[1, 0.9, 0.4], [0.9, 1, 0.01], [0.4, 0.01, 1]],
        columns=names,
        index=names,
    )
    n_draws = 1500

    ee_sobol = elementary_effects(simple_linear_model, params, cov, n_draws, "sobol")

    ee_random = elementary_effects(simple_linear_model, params, cov, n_draws, "random")

    assert_allclose(ee_sobol["mu_ind"], ee_random["mu_ind"], rtol=0.07)

    assert_allclose(ee_sobol["mu_corr"], ee_random["mu_corr"], rtol=0.07)


def test_linear_function_a():
    """Test case 1.a
    This test case is taken from Ge, Q. & M. Menendez. 2017. Extending Morris
    method for qualitative global sensitivity analysis of models with dependent
    inputs. Reliability Engineering and System Safety 162 (2017) 28–39"""

    n_inputs = 3
    names = ["x1", "x2", "x3"]
    params = pd.DataFrame(columns=["value"], data=np.zeros(n_inputs), index=names)
    cov = pd.DataFrame(
        data=[[1, 0.9, 0.4], [0.9, 1, 0.01], [0.4, 0.01, 1]],
        columns=names,
        index=names,
    )
    n_draws = 100

    ee = elementary_effects(simple_linear_model, params, cov, n_draws)

    # In paper only plots are given, no exact values. Therefore assert ranking only.
    # Assert mu_ind
    assert ee["mu_ind"].loc["x1"] < ee["mu_ind"].loc["x2"] < ee["mu_ind"].loc["x3"]

    # Assert sigma_ind
    assert ee["sigma_ind"].loc["x2"] < ee["sigma_ind"].loc["x3"]
    assert ee["sigma_ind"].loc["x1"] < ee["sigma_ind"].loc["x3"]

    # Assert mu_corr
    assert ee["mu_corr"].loc["x3"] < ee["mu_corr"].loc["x2"] < ee["mu_corr"].loc["x1"]

    # Assert sigma_corr
    assert ee["sigma_corr"].loc["x3"] < ee["sigma_corr"].loc["x2"] < ee["sigma_corr"].loc["x1"]


def test_different_seed_linear_function_a():
    """Test case 1.a
    This test case is taken from Ge, Q. & M. Menendez. 2017. Extending Morris
    method for qualitative global sensitivity analysis of models with dependent
    inputs. Reliability Engineering and System Safety 162 (2017) 28–39"""

    n_inputs = 3
    names = ["x1", "x2", "x3"]
    params = pd.DataFrame(columns=["value"], data=np.zeros(n_inputs), index=names)
    cov = pd.DataFrame(
        data=[[1, 0.9, 0.4], [0.9, 1, 0.01], [0.4, 0.01, 1]],
        columns=names,
        index=names,
    )
    n_draws = 1000

    sampling_scheme = "sobol"
    n_cores = 1
    seed = 67

    ee = elementary_effects(
        simple_linear_model,
        params,
        cov,
        n_draws,
        sampling_scheme,
        n_cores,
        seed,
    )

    # In paper only plots are given, no exact values. Therefore assert ranking only.
    # Assert mu_ind
    assert ee["mu_ind"].loc["x1"] < ee["mu_ind"].loc["x2"] < ee["mu_ind"].loc["x3"]

    # Assert sigma_ind
    assert ee["sigma_ind"].loc["x2"] < ee["sigma_ind"].loc["x3"]
    assert ee["sigma_ind"].loc["x1"] < ee["sigma_ind"].loc["x3"]

    # Assert mu_corr
    assert ee["mu_corr"].loc["x3"] < ee["mu_corr"].loc["x2"] < ee["mu_corr"].loc["x1"]

    # Assert sigma_corr
    assert ee["sigma_corr"].loc["x3"] < ee["sigma_corr"].loc["x2"] < ee["sigma_corr"].loc["x1"]


def test_linear_function_b():
    """Test case 1.b
    This test case is taken from Ge, Q. & M. Menendez. 2017. Extending Morris
    method for qualitative global sensitivity analysis of models with dependent
    inputs. Reliability Engineering and System Safety 162 (2017) 28–39"""

    n_inputs = 3
    names = ["x1", "x2", "x3"]
    params = pd.DataFrame(columns=["value"], data=np.zeros(n_inputs), index=names)
    cov = pd.DataFrame(
        data=[[1, -0.9, -0.4], [-0.9, 1, 0.01], [-0.4, 0.01, 1]],
        columns=names,
        index=names,
    )
    n_draws = 100

    ee = elementary_effects(simple_linear_model, params, cov, n_draws)

    # In paper only plots are given, no exact values. Therefore assert ranking only.
    # Assert mu_ind
    assert ee["mu_ind"].loc["x1"] < ee["mu_ind"].loc["x2"] < ee["mu_ind"].loc["x3"]

    # Assert sigma_ind (Values for x1 and x2 are very close in paper.)
    assert ee["sigma_ind"].loc["x2"] < ee["sigma_ind"].loc["x3"]
    assert ee["sigma_ind"].loc["x1"] < ee["sigma_ind"].loc["x3"]

    # Assert mu_corr
    assert ee["mu_corr"].loc["x2"] < ee["mu_corr"].loc["x1"] < ee["mu_corr"].loc["x3"]

    # Assert sigma_corr
    assert ee["sigma_corr"].loc["x2"] < ee["sigma_corr"].loc["x1"] < ee["sigma_corr"].loc["x3"]
