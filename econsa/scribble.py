import numpy as np
import pytest

from econsa.kucherenko import _general_sobol_indices
from econsa.kucherenko import _kucherenko_samples

if __name__ == "__main__":

    mean = np.array([-1, 0, 1])
    cov = np.arange(1, 4) * np.eye(3)

    def test_fun1(args):
        return np.sum(args, axis=1)

    pytest.set_trace()

    samples = _kucherenko_samples(
        mean, cov, n_draws=100_000, sampling_scheme="sobol", seed=1, skip=10_000,
    )

    first_order, total = _general_sobol_indices(test_fun1, samples, 0)
