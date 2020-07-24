from econsa.morris import _shift_cov
from econsa.morris import _shift_sample
from econsa.morris import _uniform_to_standard_normal
from econsa.morris import elementary_effects

import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal as aaae


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
