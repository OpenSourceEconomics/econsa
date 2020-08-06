"""Functions for computation of global sensitivity indices with dependent variables.

This module implements functionalities described by S. Kucherenko, S. Tarantola,
P. Annoni in 'Estimation of global sensitivity indices for models with dependent
variables' --(Comput. Phys. Commun., 183 (4) (2012), pp. 937-946)

References to Tables, Equations, etc. correspond to references in the paper mentioned
above. Variable names resemble variable names in the paper or try to be self-
explainatory.

TODO:
    - Add possibility that input data is a pandas data frame
    - Evaluate function calls outside
    - Add bounds
    - Check why sometimes errors and sometimes NaN when bounds violated
    - Pass sampler and make defualt sampler accept uniform and normal

"""
import warnings
from collections import namedtuple

import chaospy as cp
import joblib
import numba as nb
import numpy as np
import pandas as pd
from joblib import delayed
from joblib import Parallel
from scipy.stats import norm


def kucherenko_indices(
    func,
    sampling_mean,
    sampling_cov,
    n_draws=10_000,
    sampling_scheme="sobol",
    n_jobs=1,
    parallel_backend="loky",
    skip=0,
    seed_list=None,
):
    """Compute Kucherenko indices.

    Kucherenko indices aim to describe the importance of inputs on the variability
    of the output of some function. Usually one assumes that a parameter vector has been
    estimated on the input space. Most often, estimation procedures provide not only
    a point estimate but a complete distribution of the estimate. In the case of maximum
    likelihood this is for example the asymptotic normal distribution, which is
    parameterized over the mean and covariance. Here we describe the distribution of the
    input parameters over its ``sampling_mean`` and ``sampling_cov``.

    Args:
        func (callable): Function whose input-output relation we wish to analyze using
            the Kucherenko indices. Must be broadcastable.
        sampling_mean (np.ndarray): Expected value of the distribution on the input
            space.
        sampling_cov (np.ndarray): Covariance of the distribution on the input space.
        sampling_scheme (str): Sampling scheme that is used for the creation of a base
            uniform sequence from which the multivariate normal Monte Carlo sequence is
            drawn. Options are "random" and "sobol". Default is "sobol", which creates a
            Quasi Monte Carlo sequence that has favorable properties in lower
            dimensions; however if the number of parameters (``len(mean)``) exceeds ~20
            "random" can start to perform better. See https://tinyurl.com/p6grk3j.
        n_draws (int): Number of Monte Carlo draws for the estimation of the indices.
            Default is 10_000.
        n_jobs (int): Number of jobs to use for parallelization using ``joblib``.
            Default is 1.
        parallel_backend (str): Backend which will be used by ``joblib``.
        skip (int): How many values to skip of the Sobol sequence. Default is 0.
        seed_list (list or tuple): List-like object of the same length as
            ``sampling_mean`` containing (integer) seeds for the random number generator
            that will be evaluated before the sampling step for each variable. If set to
            None, range(len(``sampling_mean``)) will be used. Default is None.

    Returns:
        df_indices (pd.DataFrame): Data frame containing first_order and total order
            Kucherenko indices.

    """
    assert_input_kucherenko_indices(
        func,
        sampling_mean,
        sampling_cov,
        n_draws=10_000,
        sampling_scheme="sobol",
        n_jobs=1,
        parallel_backend="loky",
        skip=0,
        seed_list=None,
    )
    n_params = len(sampling_mean)
    seed_list = seed_list if seed_list is not None else range(n_params)

    # parallelize computation
    kwargs = {
        "func": func,
        "sampling_mean": sampling_mean,
        "sampling_cov": sampling_cov,
        "n_draws": n_draws,
        "sampling_scheme": sampling_scheme,
        "skip": skip,
    }
    with joblib.parallel_backend(backend=parallel_backend, n_jobs=n_jobs):
        indices = Parallel()(
            delayed(_kucherenko_indices_single_variable)(k, seed, **kwargs)
            for k, seed in zip(range(n_params), seed_list)
        )

    # store results
    df_indices = (
        pd.DataFrame(indices)
        .melt(
            id_vars="var",
            value_vars=["first_order", "total"],
            var_name="type",
            value_name="value",
        )
        .set_index(["var", "type"])
        .sort_index()
    )
    return df_indices


def _kucherenko_indices_single_variable(
    k, seed, func, sampling_mean, sampling_cov, n_draws, sampling_scheme, skip,
):
    """Compute Kucherenko indices for the k-th variable.

    Args:
        k (int): Variable index for which the Sobol indices should be computed.
        seed (int): Random number generator seed.
        func (callable): Function whose input-output relation we wish to analyze using
            the Kucherenko indices. Must be broadcastable.
        sampling_mean (np.ndarray): Expected value of the distribution on the input
            space.
        sampling_cov (np.ndarray): Covariance of the distribution on the input space.
        sampling_scheme (str): Sampling scheme that is used for the creation of a base
            uniform sequence from which the multivariate normal Monte Carlo sequence is
            drawn. Options are "random" and "sobol". Default is "sobol", which creates a
            Quasi Monte Carlo sequence that has favorable properties in lower
            dimensions; however if the number of parameters (``len(mean)``) exceeds ~20
            "random" can start to perform better. See https://tinyurl.com/p6grk3j.
        n_draws (int): Number of Monte Carlo draws for the estimation of the indices.
            Default is 10_000.
        skip (int): How many values to skip of the Sobol sequence. Default is 0.

    Returns:
        indices (dict): Resulting Kucherenko/Sobol indices for the k-th variable,
            stored in a dictionary with keys "var" (representing the k-th variable) and
            "first_order" and "total" denoting the first_order and total Sobol indices.

    """
    shifted_cov = _shift_cov(sampling_cov, k)
    shifted_mean = _shift_mean(sampling_mean, k)

    shifted_samples = _kucherenko_samples(
        shifted_mean, shifted_cov, n_draws, sampling_scheme, seed, skip,
    )
    first_order, total = _general_sobol_indices(func, shifted_samples, k)

    indices = {"var": k, "first_order": first_order, "total": total}
    return indices


def _general_sobol_indices(func, shifted_samples, k=0):
    """Compute general Sobol indices.

    Computes general Sobol indices for the k-th variable with the function ``func``
    given samples in ``samples``. Note that the samples are assumed to be shifted s.t.
    shifted_samples = [x_k, x_k+1, ..., x_n, x_1, ..., x_k-1], where the subindex
    denotes the index of the variable.

    Args:
        func (callable): Function whose input-output relation we wish to analyze using
            the Kucherenko indices. Must be broadcastable.
        shifted_samples (namedtuple): Namedtuple which stores the independent and
            conditional Kucherenko samples. Samples are stored under attribute names
            'independent' and 'conditional', resp. Samples are ordered such that the
            ``k``-th variable is in front.
        k (int): Variable index for which the Sobol indices should be computed.

    Returns:
        first_order, total (float): First order and total order general sobol indices.

    """
    shifted_indep = shifted_samples.independent
    shifted_cond = shifted_samples.conditional

    y_zc = np.concatenate((shifted_indep[:, :1], shifted_cond[:, 1:]), axis=1)
    yc_z = np.concatenate((shifted_cond[:, :1], shifted_indep[:, 1:]), axis=1)

    independent = _unshift_array(shifted_indep, k)
    y_zc = _unshift_array(y_zc, k)
    yc_z = _unshift_array(yc_z, k)

    f_y_z = func(independent)

    f_y_zc = func(y_zc)
    f_yc_z = func(yc_z)

    mean_sq_yz = np.mean(f_y_z ** 2)
    sq_mean_yz = np.mean(f_y_z) ** 2
    D = mean_sq_yz - sq_mean_yz

    # Equation 5.3
    first_order = (np.mean(f_y_z * f_y_zc) - sq_mean_yz) / D

    # Equation 5.4
    total = np.mean((f_y_z - f_yc_z) ** 2) / (2 * D)

    return first_order, total


def _kucherenko_samples(mean, cov, n_draws, sampling_scheme, seed, skip):
    """Draw samples from independent and conditional distribution.

    Draw samples as formulated in the second algorithm of [Section 6]. Steps of the
    algorithm [a) - i)] are marked as comments in the code. Variable names should
    coincide with symbols used in Kucherenko et al. 2012.

    Args:
        mean (np.ndarray): Array of shape d x 1, representing the mean of the
            distribution of which we want to sample.
        cov (np.ndarray): Array of shape d x d, representing the covariance of
            the distribution of which we want to sample.
        n_draws (int): Number of samples to draw.
        sampling_scheme (str): One of ["sobol", "random"].
        seed (int): Random number generator seed.
        skip (int): How many values to skip of the Sobol sequence.

    Returns:
        samples (namedtuple): Namedtuple which stores the independent and conditional
            samples stored under attribute 'independent' and 'conditional', resp.

    """
    n_params = len(mean)  # In the paper this variable is referred to as "n".

    # a) Draw uniform distributed (base) samples
    u, u_prime = _get_uniform_base_draws(n_draws, n_params, sampling_scheme, seed, skip)

    # b) with s = 1. Split uniform sample in two groups.
    v_prime = u_prime[:, :1]
    w_prime = u_prime[:, 1:]

    # c) Transform uniform draws to multivariate normal draws.
    x = _uniform_to_multivariate_normal(u, mean, cov)

    # d) Split multivariate normal sample in two groups.
    y = x[:, :1]
    z = x[:, 1:]

    # e)
    z_tilde = _uniform_to_standard_normal(w_prime)
    y_tilde = _uniform_to_standard_normal(v_prime)

    # Auxiliary step.
    mean_y = mean[:1]
    mean_z = mean[1:]
    cov_y = cov[:1, :1]
    cov_z = cov[1:, 1:]
    cov_yz = cov[1:, 0].reshape(n_params - 1, 1)

    # f) Compute the conditional mean.
    mean_z_given_y = _conditional_mean(y, mean_z, mean_y, cov_y, cov_yz)
    mean_y_given_z = _conditional_mean(z, mean_y, mean_z, cov_z, cov_yz.T)

    # g) Compute the conditional covariance.
    cov_z_given_y = _conditional_covariance(cov_z, cov_y, cov_yz)
    cov_y_given_z = _conditional_covariance(cov_y, cov_z, cov_yz.T)

    # h)
    z_bar = _standard_normal_to_multivariate_normal(
        z_tilde, mean_z_given_y, cov_z_given_y,
    )
    y_bar = _standard_normal_to_multivariate_normal(
        y_tilde, mean_y_given_z, cov_y_given_z,
    )

    # i) Combine results.
    independent = np.hstack([y, z])
    conditional = np.hstack([y_bar, z_bar])

    samples = namedtuple("Samples", "independent conditional")(
        independent=independent, conditional=conditional,
    )
    return samples


def _conditional_mean(y, mean_z, mean_y, cov_y, cov_yz):
    """Compute conditional mean of normal marginal.

    Compute conditional mean of variable z given a realization of variable y. See either
    Kucherenko et al. 2012 [Equation 3.4] or https://tinyurl.com/jbsrcue.

    For the below example consider the case of two variables z and y. Let the expected
    value be E([z, y]) = [-1, 2] and the covariance Cov([z, y]) = [[1, 0.5], [0.5, 2]].
    Applying the formula for conditional expectation for normal variables we then get

            E(z|y=y) = E(z) + cov(z, y) * (1 / var(y)) * (y - E(y))
                     = E(z) + (1 / 4) * (y - E(y))
                     = -1 + (1/4) * (y - 2)

    Args:
        y (np.ndarray): Array on which to condition on.
        mean_z (np.ndarray): Mean of variable `z`.
        mean_y (np.ndarray): Mean of variable `y`.
        cov_y (np.ndarray): Covariance-variance matrix of variable `y`.
        cov_yz (np.ndarray): Covariance of variables `z` and `y`.

    Returns:
        mean_z_given_y (np.ndarray): Conditional mean of `z` given the realization y.


    """
    update = cov_yz.dot(np.linalg.inv(cov_y)).dot((y - mean_y).T).T
    mean_z_given_y = mean_z + update
    return mean_z_given_y


def _conditional_covariance(cov_z, cov_y, cov_zy):
    """Compute conditional covariance of normal marginal.

    Compute conditional covariance of variable z given variable y. See either
    Kucherenko et al. 2012 [Equation 3.5] or https://tinyurl.com/jbsrcue.

    For the below example consider the case of two variables z and y. Let the expected
    value be E([z, y]) = [-1, 2] and the covariance Cov([z, y]) = [[1, 0.5], [0.5, 2]].
    Applying the formula for conditional covariance for normal variables we then get

            Var(z|y=y) = var(z) - cov(z, y) * (1 / var(y)) * cov(y, z)
                       = var(z) - cov(z, y) ** 2 / var(y)
                       = 1 - (0.5) ** 2 / 2
                       = 0.875

    Args:
        cov_z (np.ndarray): Variance-Covariance matrix of variable `z`.
        cov_y (np.ndarray): Variance-Covariance matrix of variable `y`.
        cov_zy (np.ndarray): Covariance of variables `z` and `y`.

    Returns:
        cov_z_given_y (np.ndarray): Conditional covariance of `z` given `y`.

    """
    update = cov_zy.dot(np.linalg.inv(cov_y)).dot(cov_zy.T)
    cov_z_given_y = cov_z - update
    return cov_z_given_y


def _get_uniform_base_draws(n_draws, n_params, sampling_scheme, seed=0, skip=0):
    """Get uniform random draws.

    TODO:
        - Should we replace random by quasi-random?
        - Should we de-correlate the result as a finite sample correction.

    Args:
        n_draws (int): Number of uniform draws to generate.
        n_params (int): Number of parameters of model.
        sampling_scheme (str): one of ["sobol", "random"]
        seed (int): Random number generator seed; default is 0.
        skip (int): How many values to skip of the Sobol sequence.

    Returns:
        u, u_prime (np.ndarray): Arrays of shape (n_draws, n_params-1) and (n_draws,1)
            with i.i.d draws from a uniform [0, 1] distribution.

    """
    np.random.seed(seed)

    if sampling_scheme == "sobol":
        draws = cp.generate_samples(
            order=n_draws + skip, domain=2 * n_params, rule="S",
        ).T
    elif sampling_scheme == "random":
        draws = np.random.uniform(size=(n_draws, 2 * n_params))
        # draws = np.random.uniform(low=1e-5, high=1-1e-5, size=(n_draws, 2 * n_params))
    else:
        raise ValueError("Argument 'sampling_scheme' is not in {'sobol', 'random'}.")

    skip = skip if sampling_scheme == "sobol" else 0

    u = draws[skip:, :n_params]
    u_prime = draws[skip:, n_params:]
    return u, u_prime


def _uniform_to_standard_normal(uniform):
    """Convert i.i.d uniform draws to i.i.d standard normal draws.

    Args:
        uniform (np.ndarray): Can have any shape.

    Returns
        standard_normal (np.ndarray): Same shape as uniform.

    """
    standard_normal = norm.ppf(uniform)
    return standard_normal


def _standard_normal_to_multivariate_normal(draws, mean, cov):
    """Transform standard normal draws to multivariate normal.

    Args:
        draws (np.ndarray): Draws from a standard normal distribution. Shape has the
            form (n_draws, n_params).
        mean (np.ndarray): Mean of the new distribution. Must have length n_params.
        cov (np.ndarray): Covariance of the new distribution. Must have shape
            (n_params, n_params).

    Returns:
        multivariate_draws (np.ndarray): Draws from the multivariate normal. Shape has
            the form (n_draws, n_params).

    """
    cholesky = np.linalg.cholesky(cov)
    multivariate_draws = mean + cholesky.dot(draws.T).T
    return multivariate_draws


def _uniform_to_multivariate_normal(uniform, mean, cov):
    """Transform uniform draws to multivariate normal.

    Args:
        uniform (np.ndarray): Uniform draws with shape (n_draws, n_params).
        mean (np.ndarray): Mean of the new distribution. Must have length n_params.
        cov (np.ndarray): Covariance of the new distribution. Must have shape
            (n_params, n_params).

    Returns:
        normal (np.ndarray): Draws from the multivariate normal. Shape has the form
            (n_draws, n_params).

    """
    cov = np.atleast_2d(cov)
    standard_normal = _uniform_to_standard_normal(uniform)
    normal = _standard_normal_to_multivariate_normal(standard_normal, mean, cov)
    return normal


@nb.guvectorize(
    ["f8[:], i8, f8[:]", "i8[:], i8, i8[:]"], "(m), () -> (m)", nopython=True,
)
def _shift_sample(sample, k, out):
    """Re-sort sample such that the first k elements are moved to the end.

    If sample is multidimensional this works on the last axis.

    This corresponds to the function tau_1 in section 3.3 of GM17 but can also
    be used as tau_3 if called with a different k.

    guvectorize is not used for speed, but to get automatic broadcasting.

    Args:
        sample (np.ndarray): Array of shape [..., n_params]
        k (int): 0 <= k <= n_params

    Returns:
        shifted (np.ndarray): Same shape as sample.

    """
    n_params = len(sample)
    for old_pos in range(k):
        new_pos = n_params - k + old_pos
        out[new_pos] = sample[old_pos]

    for new_pos, old_pos in enumerate(range(k, n_params)):
        out[new_pos] = sample[old_pos]


def _shift_cov(cov, k):
    """Re-sort a covariance matrix such that the fist k elements are moved to the end.

    Args:
        cov (np.ndarray): Two dimensional array of shape (n_params, n_params)
        k (int): 0 <= k <= n_params

    Returns:
        shifted (np.ndarray): Same shape as cov.

    """
    n_params = len(cov)
    old_order = np.arange(n_params).astype(int)
    new_order = _shift_sample(old_order, k).astype(int)

    shifted = cov.copy()[new_order][:, new_order]
    return shifted


def _unshift_cov(cov, k):
    """Re-sort a covariance matrix such that the last k elements are moved to the start.

    Args:
        cov (np.ndarray): Two dimensional array of shape (n_params, n_params)
        k (int): 0 <= k <= n_params

    Returns:
        unshifted (np.ndarray): Same shape as cov.

    """
    n_params = len(cov)
    unshifted = _shift_cov(cov, n_params - k)
    return unshifted


def _shift_mean(mean, k):
    """Re-sort a mean vector such that the fist k elements are moved to the end.

    Args:
        mean (np.ndarray): One dimensional array of shape (n_params).
        k (int): 0 <= k <= n_params.

    Returns:
        shifted (np.ndarray): Same shape as mean.

    """
    n_params = len(mean)
    old_order = np.arange(n_params).astype(int)
    new_order = _shift_sample(old_order, k).astype(int)

    shifted = mean.copy()[new_order]
    return shifted


def _unshift_mean(mean, k):
    """Re-sort a mean vector such that the fist k elements are moved to the end.

    Args:
        mean (np.ndarray): One dimensional array of shape (n_params).
        k (int): 0 <= k <= n_params.

    Returns:
        unshifted (np.ndarray): Same shape as mean.

    """
    n_params = len(mean)
    unshifted = _shift_mean(mean, n_params - k)
    return unshifted


def _shift_array(arr, k):
    """Re-sort a 2d array such that the fist k columns are moved to the end.

    Args:
        arr (np.ndarray): Two dimensional array of shape (n_draws, n_params).
        k (int): 0 <= k <= n_params.

    Returns:
        shifted (np.ndarray): Same shape as array.

    """
    n_params = arr.shape[1]

    old_order = np.arange(n_params).astype(int)
    new_order = _shift_sample(old_order, k).astype(int)

    shifted = arr.copy()[:, new_order]
    return shifted


def _unshift_array(arr, k):
    """Re-sort a 2d array such that the last k columns are moved to the beginning.

    If ``arr`` was produced by calling ``_shift_array`` on some original array arr_0
    with shift k_0, then calling ``_unshift_array`` on ``arr``, with ``k`` equal to
    the number of columns of ``arr`` minus k_0, recovers the original array arr_0.

    Args:
        arr (np.ndarray): Two dimensional array of shape (n_draws, n_params).
        k (int): 0 <= k <= n_params.

    Returns:
        unshifted (np.ndarray): Same shape as array.

    """
    n_params = arr.shape[1]
    unshifted = _shift_array(arr, n_params - k)
    return unshifted


def assert_input_kucherenko_indices(
    func,
    sampling_mean,
    sampling_cov,
    n_draws,
    sampling_scheme,
    n_jobs,
    parallel_backend,
    skip,
    seed_list,
):
    n_params = len(sampling_mean)

    assert sampling_cov.shape == (n_params, n_params), (
        "Argument 'sampling_cov' does not have a compatible dimension with argument "
        "'sampling_mean'."
    )
    assert (
        isinstance(n_draws, int) and n_draws > 0
    ), "Argument 'n_draws' must be a positive integer."
    assert sampling_scheme in {
        "random",
        "sobol",
    }, "Argument 'sampling_scheme' must be in {'random', 'sobol'}."
    assert (
        isinstance(n_jobs, int) and n_jobs >= 1
    ), "Argument 'n_jobs' must be a positive integer."

    if seed_list is not None and len(seed_list) != n_params:
        warnings.warn(
            "Argument 'seed_list' does not has the same length as argument"
            "'sampling_mean'; Using seed_list = range(len(sampling_mean)).",
            UserWarning,
        )
