import chaospy as cp
import numpy as np

from econsa.kucherenko import _conditional_covariance
from econsa.kucherenko import _conditional_mean
from econsa.kucherenko import _shift_cov
from econsa.kucherenko import _shift_mean
from econsa.kucherenko import _uniform_to_multivariate_normal
from econsa.kucherenko import _unshift_mean


def kucherenko_index(
    func,
    mean,
    cov,
    dimensions=None,
    n_joint=10_000,
    n_outer=100,
    n_inner=100,
    scheme="random",
    seed=0,
):
    """Kucherenko indices wrapper function."""
    if dimensions is None:
        dimensions = list(range(len(mean)))

    indices = {}
    for dimension in dimensions:
        index = _kucherenko_index(
            func, dimension, mean, cov, scheme, n_joint, n_outer, n_inner, seed,
        )
        indices[dimension] = index
    return indices


def _kucherenko_index(
    func, dimension, mean, cov, scheme, n_joint, n_outer, n_inner, seed,
):
    """Kucherenko index inner functions."""
    # draw (independent) samples100100100
    joint_samples = _create_joint_samples(n_joint, mean, cov, scheme, seed)

    # approximate mean and variance (f_0 and D in paper)
    func_mean, evals = _compute_mean(func, joint_samples, return_evals=True)
    func_variance = _compute_variance(func, joint_samples, func_mean, evals)

    # approximate double integral
    double_integral = 0
    for i in range(n_outer):
        outer_sample = _create_outer_sample(dimension, mean, cov, scheme, seed + i)

        inner_integral = 0
        for j in range(n_inner):
            inner_sample = _create_inner_sample(
                outer_sample, dimension, mean, cov, scheme, seed + j,
            )
            x = _combine_samples(outer_sample, inner_sample, dimension)
            inner_integral += func(x)

        double_integral += (inner_integral / n_inner) ** 2

    index = (double_integral / n_outer - func_mean ** 2) / func_variance
    return index


def _compute_mean(func, samples, return_evals=False):
    """Compute mean of func using samples."""
    evaluations = _evaluate_func(func, samples)
    mean = evaluations.mean()

    out = (mean, evaluations) if return_evals else mean
    return out


def _compute_variance(func, samples, mean=None, evaluations=None):
    """Compute variance of func using samples."""
    if mean is None:
        mean = _compute_mean(func, samples)

    if evaluations is None:
        evaluations = _evaluate_func(func, samples)

    variance = (evaluations ** 2).mean() - mean ** 2
    return variance


def _combine_samples(outer_sample, inner_sample, dimension, unshift=True):
    """Combine outer and inner samples.

    Args:
        outer_sample (float): Value to sort in.
        inner_sample (np.ndarray): 1d numpy array
        dimension (int): Index where to sort in value.
        unshift (bool): If the result should be unshifted.

    Returns:
        sample (np.ndarray): The combined sample.

    """
    sample = np.insert(inner_sample, dimension, outer_sample)
    if unshift:
        sample = _unshift_mean(sample, dimension)
    return sample


def _evaluate_func(func, samples):
    """Evaluate func on various types of samples.

    Args:
        func (callable): Function.
        samples (np.ndarray): 2d array with samples to evaluate.

    Returns:
        evaluatiosn (np.ndarrary): 1d array with evaluations.

    """
    evaluations = np.array([func(sample) for sample in samples])
    return evaluations


def _create_joint_samples(n_joint, mean, cov, scheme, seed):
    """Draw from joint distribution of features."""
    uniform_draws = _get_uniform_base_draws(n_joint, len(mean), scheme, seed)
    joint_sample = _uniform_to_multivariate_normal(uniform_draws, mean, cov)
    return joint_sample


def _create_outer_sample(dimension, mean, cov, scheme, seed):
    """Draw from single feature dimension."""
    uniform_draws = _get_uniform_base_draws(1, 1, scheme, seed)
    mean = np.atleast_1d(mean[dimension])
    var = np.atleast_2d(cov[dimension, dimension])
    outer_samples = _uniform_to_multivariate_normal(uniform_draws, mean, var)
    return outer_samples


def _create_inner_sample(outer_sample, dimension, mean, cov, scheme, seed):
    """Draw from parameter vector conditional on single dimension."""
    mean, cov = _conditional_mean_and_cov(outer_sample, dimension, mean, cov)

    uniform_draws = _get_uniform_base_draws(1, len(mean), scheme, seed)
    inner_sample = _uniform_to_multivariate_normal(uniform_draws, mean, cov).flatten()
    inner_sample = _unshift_mean(inner_sample, dimension)
    return inner_sample


def _get_uniform_base_draws(n_draws, n_params, sampling_scheme, seed=0):
    """Get uniform random draws."""
    np.random.seed(seed)

    if sampling_scheme == "sobol":
        draws = cp.generate_samples(order=n_draws, domain=n_params, rule="S").T
    elif sampling_scheme == "random":
        draws = np.random.uniform(size=(n_draws, n_params))
    else:
        raise ValueError("Argument 'sampling_scheme' is not in {'sobol', 'random'}.")

    return draws


def _conditional_mean_and_cov(outer_sample, dimension, mean, cov):
    """Return conditional mean and covariance."""
    mean = _shift_mean(mean.copy(), dimension)
    cov = _shift_cov(cov.copy(), dimension)

    mean_y = mean[:1]
    mean_z = mean[1:]
    cov_y = cov[:1, :1]
    cov_z = cov[1:, 1:]
    cov_yz = cov[1:, 0].reshape(len(mean) - 1, 1)

    mean = _conditional_mean(outer_sample, mean_z, mean_y, cov_y, cov_yz).flatten()
    cov = _conditional_covariance(cov_z, cov_y, cov_yz)
    return mean, cov
