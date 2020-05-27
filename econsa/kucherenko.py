"""Functions for computation of global sensitivity indices with dependent variables.

This module implements functionalities described by S. Kucherenko, S. Tarantola,
P. Annoni in "Estimation of global sensitivity indices for models with dependent
variables" --(Comput. Phys. Commun., 183 (4) (2012), pp. 937-946)

References to Tables, Equations, etc. correspond to references in the paper mentioned
above.
TODO:
    - Parameters in data frame (value column is numpy array)
    - Rewrite code from module aleeciu to adhere to our standards
"""
from collections import namedtuple

import chaospy as cp
import numba as nb
import numpy as np
from scipy.stats import norm


def kucherenko_indices(
    func, sampling_mean, sampling_cov, n_draws=10_000, sampling_scheme="sobol", n_jobs=1
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
            the Kucherenko indices.
        sampling_mean: Expected value of the distribution on the input space.
        sampling_cov: Covariance of the distribution on the input space.
        sampling_scheme: Sampling scheme that is used for the creation of a base uniform
            sequence from which the multivariate normal Monte Carlo sequence is drawn.
            Options are "random" and "sobol". Default is "sobol", which creates a Quasi
            Monte Carlo sequence that has favorable properties in lower dimensions;
            however if the number of parameters (``len(mean)``) exceeds ~20 "random"
            can start to perform better. See https://tinyurl.com/p6grk3j.
        n_draws: Number of Monte Carlo draws for the estimation of the indices. Default
            is 10_000.
        n_jobs (int): Number of jobs to use for parallelization using ``joblib``.
            Default is 1.

    Returns:
        out (list): List of tuples.

    """
    n_params = len(sampling_mean)

    shifted_cov = sampling_cov.copy()
    shifted_mean = sampling_mean.copy()

    out = []
    for _ in range(n_params):
        samples = _kucherenko_samples(
            shifted_mean, shifted_cov, n_draws, sampling_scheme
        )
        first_order, total = _general_sobol_indices(func, samples)

        shifted_cov = _shift_cov(shifted_cov, 1)
        shifted_mean = _shift_mean(shifted_mean, 1)

        out.append((first_order, total))

    return out


def _general_sobol_indices(func, samples, *args, **kwargs):
    """Compute general Sobol indices.

    Computes general Sobol indices using

    Args:
        func
        samples:
        *args:
        **kwargs:

    Returns:

    """
    return None, None


def _kucherenko_samples(mean, cov, n_draws, sampling_scheme):
    """Draw samples from independent and conditional distribution.

    TODO:
        - FIND OUT IF THIS IS IMPORTANT: cov_new = np.cov(x)
        - Explain better what I'm doing here
        - Name, kuchereno_samples or conditional_gaussian_sampling?

    Draw samples as formulated in the second algorithm of [Section 6]. Steps of the
    algorithm [a) - i)] are marked as comments in the code.


    Args:
        mean (np.ndarray): Array of shape d x 1, representing the mean of the
            distribution of which we want to sample.
        cov (np.ndarray): Array of shape d x d, representing the covariance of
            the distribution of which we want to sample.
        n_draws (int):
        sampling_scheme (str): one of ["sobol", "random"]
        *args:
        **kwargs:

    Returns:
        samples (namedtuple): Namedtuple which stores the independent and conditional
            samples stored under attribute 'independent' and 'conditional', resp.

    """
    n_params = len(mean)  # In the paper this variable is referred to as "n".

    # a)
    u, u_prime = _get_uniform_base_draws(n_draws, n_params, sampling_scheme)

    # b) with s = 1.
    v_prime = u_prime[:, :1]
    w_prime = u_prime[:, 1:]

    # c) Transform uniform draws to multivariate normal draws.
    u_normal = _uniform_to_standard_normal(u)
    x = _standard_normal_to_multivariate_normal(u_normal, mean, cov)

    # d)
    y = x.T[:, :1]
    z = x.T[:, 1:]

    # e)
    z_tilde = _uniform_to_standard_normal(w_prime)
    y_tilde = _uniform_to_standard_normal(v_prime)

    # Auxiliary step.
    mean_y = mean[:1]
    mean_z = mean[1:]
    cov_y = cov[:1, :1]
    cov_z = cov[1:, 1:]
    cov_zy = cov[1:, 0]

    # f) Compute the conditional mean.
    mean_z_given_y = _conditional_mean(y, mean_z, mean_y, cov_y, cov_zy)
    mean_y_given_z = _conditional_mean(z, mean_y, mean_z, cov_z, cov_zy.T)

    # g) Compute the conditional covariance.
    cov_z_given_y = _conditional_covariance(cov_z, cov_y, cov_zy)
    cov_y_given_z = _conditional_covariance(cov_y, cov_z, cov_zy.T)

    # h)
    z_bar = _standard_normal_to_multivariate_normal(
        z_tilde, mean_z_given_y, cov_z_given_y
    )
    y_bar = _standard_normal_to_multivariate_normal(
        y_tilde, mean_y_given_z, cov_y_given_z
    )

    # i) Combine results.
    independent = np.hstack([y, z])
    conditional = np.hstack([y_bar, z_bar])

    samples = namedtuple("Samples", "independent conditional")(
        independent=independent, conditional=conditional
    )
    return samples


def _standard_normal_to_multivariate_normal(draws, mean, cov):
    """Transform standard normal draws to multivariate normal.

    Args:
        draws (np.ndarray): Draws from a standard normal distribution.
        mean (np.ndarray): Mean of the new distribution.
        cov (np.ndarray): Covariance of the new distribution.

    Returns:
        multivariate_draws (np.ndarray): Draws from the multivariate normal.

    """
    cholesky = np.linalg.cholesky(cov)
    multivariate_draws = mean + cholesky.dot(draws)
    return multivariate_draws


def _conditional_mean(y, mean_z, mean_y, cov_y, cov_zy):
    """Compute conditional mean of normal marginal.

    Compute conditional mean of variable z given a realization of variable y. See either
    Kucherenko et al. 2012 [Equation 3.4] or https://tinyurl.com/jbsrcue.

    Args:
        y (np.ndarray): Array on which to condition on.
        mean_z (np.ndarray): Mean of variable `z`.
        mean_y (np.ndarray): Mean of variable `y`.
        cov_y (np.ndarray): Covariance-variance matrix of variable `y`.
        cov_zy (np.ndarray): Covariance of variables `z` and `y`.

    Returns:
        mean_z_given_y (np.ndarray): Conditional mean of `z` given the realization y.

    """
    update = cov_zy.dot(np.linalg.inv(cov_y)).dot(y - mean_y)
    mean_z_given_y = mean_z + update
    return mean_z_given_y


def _conditional_covariance(cov_z, cov_y, cov_zy):
    """Compute conditional covariance of normal marginal.

    Compute conditional covariance of variable z given variable y. See either
    Kucherenko et al. 2012 [Equation 3.5] or https://tinyurl.com/jbsrcue.

    Args:
        cov_z (np.ndarray): Variance-Covariance matrix of variable `z`.
        cov_y (np.ndarray): Variance-Covariance matrix of variable `y`.
        cov_zy (np.ndarray): Covariance of variables `z` and `y`.

    Returns:
        cov_z_given_y (np.ndarray): Conditional covariance of `z` given `y`.

    """
    update = cov_zy.dot(np.linalg.inv(cov_y)).dot(cov_zy.T)
    cov_z_given_y = cov_z + update
    return cov_z_given_y


def _get_uniform_base_draws(n_draws, n_params, sampling_scheme):
    """Get uniform random draws.

    Questions:
    - Should we replace random by quasi-random?
    - Should we de-correlate the result as a finite sample correction.

    Args:
        n_draws (int): Number of uniform draws to generate.
        n_params (int): Number of parameters of model.
        sampling_scheme (str): one of ["sobol", "random"]

    Returns:
        u, u_prime (np.ndarray): Arrays of shape (n_draws, n_params) with i.i.d draws
            from a uniform [0, 1] distribution.

    """
    if sampling_scheme == "sobol":
        draws = cp.generate_samples(order=n_draws * 2 * n_params, rule="S").reshape(
            n_draws, -1
        )

    elif sampling_scheme == "random":
        draws = np.random.uniform(low=1e-5, high=1 - 1e-5, size=(n_draws, 2 * n_params))

    else:
        raise ValueError("Argument 'sampling_scheme' is not in {'sobol', 'random'}.")

    u = draws[:, :n_params]
    u_prime = draws[:, n_params:]

    return u, u_prime


def _uniform_to_standard_normal(uniform):
    """Convert i.i.d uniform draws to i.i.d standard normal draws.

    Args:
        uniform (np.ndarray): Can have any shape.

    Returns
        standard_normal (np.ndarray): Same shape as uniform.

    """
    return norm.ppf(uniform)


def _dependent_draws(z_a, z_b, mean, cov, kind):
    """Create dependent draws for EE^ind with radial design.

    This calculates the two terms of the numerator of equation 39 in GM17, i.e.
    conditions on all but one element.

    The generation of the dependent random samples is simplified because we condition
    on all but one elements. Therefore, and due to the normality assumption, converting
    the re-drawn element to the desired distribution is just rescaling with the standard
    deviation of the conditional distribution and adding the conditional mean.

    Args:
        z_a (np.ndarray): Array of shape (n_draws, n_params) with the "a-sample"
            converted to i.i.d standard normal distribution.
        z_b (np.ndarray): Array of shape (n_draws, n_params) with the "b-sample"
            converted to a i.i.d standard normal distribution.
        mean (np.ndarray): Array of length n_params
        cov (np.ndarray): Array of shape (n_params, n_params)
        kind (str): one of ["ind", "corr"]

    Returns:
        ab_sample_x (np.ndarray): Array of shape (n_draws, n_params, n_params) with the
            left term from the numerator of equation 39 in GM17.
        a_sample_x (np.ndarray): Array of shape (n_draws, n_params, n_params) with the
            right term from the numerator of equation 39 in GM17

    """
    # extract dimensions
    n_draws, n_params = z_a.shape

    if kind == "ind":
        shift = np.arange(n_params).astype(int) + 1
        diag_pos = -1
    elif kind == "corr":
        shift = np.arange(n_params).astype(int)
        diag_pos = 0
    else:
        raise ValueError

    unshift = n_params - shift

    # generate the shifted a sample
    a_sample_z = np.repeat(z_a, n_params, axis=0).reshape(n_draws, n_params, n_params)

    a_sample_z_shifted = _shift_sample(a_sample_z, shift.reshape(1, -1))

    ab_sample_z_shifted = a_sample_z_shifted.copy()
    ab_sample_z_shifted[:, :, diag_pos] = z_b

    shifted_covs = np.zeros((n_params, n_params, n_params))
    for p, s in enumerate(shift):
        shifted_covs[p] = _shift_cov(cov, s)

    # calculated shifted means (one per parameter). They are aligned with shifted covs.
    means = np.repeat(mean.reshape(1, n_params), n_params, axis=0)
    shifted_means = _shift_sample(means, np.arange(n_params, dtype=int) + 1)

    # convert the shifted a sample to a multivariate normal distribution
    shifted_chols = np.linalg.cholesky(shifted_covs)
    a_sample_x_shifted = (
        np.matmul(
            shifted_chols, a_sample_z_shifted.reshape(n_draws, n_params, n_params, 1)
        ).reshape(n_draws, n_params, n_params)
        + shifted_means
    )

    ab_sample_x_shifted = (
        np.matmul(
            shifted_chols, ab_sample_z_shifted.reshape(n_draws, n_params, n_params, 1)
        ).reshape(n_draws, n_params, n_params)
        + shifted_means
    )
    # un-shift the shifted samples.
    a_sample_x = _shift_sample(a_sample_x_shifted, unshift.reshape(1, -1))
    ab_sample_x = _shift_sample(ab_sample_x_shifted, unshift.reshape(1, -1))

    return ab_sample_x, a_sample_x


@nb.guvectorize(
    ["f8[:], i8, f8[:]", "i8[:], i8, i8[:]"], "(m), () -> (m)", nopython=True
)
def _shift_sample(sample, k, out):
    """Re-sort sample such that the first k elements are moved to the end.

    If sample is multidimensional this works on the last axis.

    This corresponds to the function tau_1 in section 3.3 of GM17 but can also
    be used as tau_3 if called with a different k.

    guvectorize is not used for speed, but to get automatic broadcasting.

    Args:
        sample (np.ndarray): Array of shape [..., m]
        k (int): 0 <= k <= m

    Returns:
        shifted (np.ndarray): Same shape as sample.

    """
    m = len(sample)
    for old_pos in range(k):
        new_pos = m - k + old_pos
        out[new_pos] = sample[old_pos]

    for new_pos, old_pos in enumerate(range(k, m)):
        out[new_pos] = sample[old_pos]


def _shift_cov(cov, k):
    """Re-sort a covariance matrix such that the fist k elements are moved to the end.

    Args:
        cov (np.ndarray): Two dimensional array of shape (m, m)
        k (int): 0 <= k <= m

    Returns:
        shifted (np.ndarray): Same shape as cov.

    """
    m = len(cov)
    old_order = np.arange(m).astype(int)
    new_order = _shift_sample(old_order, k).astype(int)
    return cov[new_order][:, new_order]


def _shift_mean(mean, k):
    """Re-sort a mean vector such that the fist k elements are moved to the end.

    Args:
        mean (np.ndarray): One dimensional array of shape (m)
        k (int): 0 <= k <= m

    Returns:
        shifted (np.ndarray): Same shape as mean.

    """
    m = len(mean)
    old_order = np.arange(m).astype(int)
    new_order = _shift_sample(old_order, k).astype(int)
    return mean[new_order]


def _marginal_mean_and_covariance(mean, cov):
    """Return marginal mean and covariance.

    Returns the marginal mean and covariance of the variable defined by the first
    element of mean and all other elements, as well as the covariance between the first
    element and all other. That is, let X=(X1,...,Xn) be a random vector with expected
    value equal to ``mean`` and covariance-variance matrix equal to ``cov``. Then we
    consider the mean and variance of Y = X1 and Z = (X2,...,Xn), as well as the
    covariance vector of Y with each element in Z.

    Args:
        mean (np.ndarray): Mean vector.
        cov (np.ndarray): Variance-Covariance matrix.

    Returns:
        mean_y, mean_z, cov_y, cov_z, cov_zy (np.ndarray): Mean of Y and Z, Variance-
            covariance matrix of Y and Z and covariance vector between Y and Z.

    """
    mean_y = mean[:1]
    mean_z = mean[1:]

    cov_y = cov[:1, :1]
    cov_z = cov[1:, 1:]

    cov_zy = cov[1:, 0]

    return mean_y, mean_z, cov_y, cov_z, cov_zy
