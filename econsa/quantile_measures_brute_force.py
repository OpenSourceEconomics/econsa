"""The brute fot estimators of quantile based global sensitivity measures.

This module contains functions to calculate global sensitivity measures based on
quantiles of the output introduced by Kucherenko et al.(2019).

"""
import chaospy as cp
import numpy as np
from numba import jit
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import uniform


def bf_mcs_quantile(
    func,
    n_params,
    loc,
    scale,
    dist_type,
    n_draws,
    sampling_scheme="sobol",
    seed=0,
    skip=0,
):
    r"""Compute (Quasi) Monte Carlo estimators of quantile based global sensitivity measures.

    This function implements the Double loop reordering
    (DLR) approach described in Section 4.1 of [K2019]_.

    Parameters
    ----------
    func : callable
        Objective function to calculate the quantile-based measures. Must be broadcastable.

    n_params : int
        Number of parameters of objective function.

    loc : float or np.ndarray
        The location(`loc`) keyword passed to `scipy.stats.norm`_ function to shift the
        location of "standardized" distribution. Specifically, for normal distribution
        it specifies the mean with the length of `n_params`.

        .. _scipy.stats.norm: https://docs.scipy.org/doc/scipy/reference/generated/
            _scipy.stats.norm.html

    scale : float or np.ndarray
        The `scale` keyword passed to `scipy.stats.norm`_ function to adjust the scale of
        "standardized" distribution. Specifically, for normal distribution it specifies
        the covariance matrix of shape (n_params, n_params).

    dist_type : str
        The distribution type of input. Options are "Normal", "Exponential" and "Uniform".

    n_draws : int
        Number of sampled points. This will later turn into the number of Monte Carlo draws.
        Accroding to [K2017]_, to preserve the uniformity properties `n_draws` should always be
        equal to :math:`n_draws = 2^p`, where :math:`p` is an integer.

    sampling_scheme : str
        Sampling scheme that is used for the creation of a base
        uniform sequence from which the multivariate normal Monte Carlo sequence is
        drawn. Options are "random" and "sobol". Default is "sobol", which creates a
        Quasi Monte Carlo sequence that has favorable properties in lower
        dimensions; however if the number of parameters (``len(mean)``) exceeds ~20
        "random" can start to perform better. See https://tinyurl.com/p6grk3j.

    seed : int
        Random number generator seed.

    skip : int
        Number of values to skip of Sobol sequence. Default is `0`.

    Returns
    -------
    q_1 : np.ndarray
        Quantile based measure. Shape has the form (len(alp), n_params).

    q_2 : np.ndarray
        Quantile based measure. Shape has the form (len(alp), n_params).

    norm_q_1 : np.ndarray
        Nomalized quantile based measure. Shape has the form (len(alp), n_params).

    norm_q_2 : np.ndarray
        Nomalized quantile based measure. Shape has the form (len(alp), n_params).

    References
    ----------
    .. [K2019] S. Kucherenko, S. Song, L. Wang. Quantile based global
        sensitivity measures, Reliab. Eng. Syst. Saf. 185 (2019) 35–48.

    .. [K2017] Kucherenko S, Song S. Different numerical estimators
        for main effect global sensitivity indices. Reliab Eng Syst
        Saf 2017;165:222–38.
    """
    # range of alpha
    dalp = (0.98 - 0.02) / 30
    alp = np.arange(0.02, 0.98 + dalp, dalp)  # len(alp) = 31

    # get the two independent sample sets from a joint PDF
    x, x_prime = _bf_unconditional_sample(
        n_draws,
        n_params,
        dist_type,
        loc,
        scale,
        sampling_scheme="sobol",
        seed=0,
        skip=0,
    )

    # get the conditional sample set
    x_mix = _bf_conditional_sample(x, x_prime)

    # quantile of output calculated with unconditional sample set
    quantile_y_x = _bf_unconditional_quantile_y(x, alp, func)

    # quantile of output calculated with conditional sample set
    quantile_y_x_mix = _bf_conditional_quantile_y(x_mix, alp, func)

    # Get quantile based measures
    q_1, q_2 = _bf_quantile_measures(quantile_y_x, quantile_y_x_mix)

    # Get normalized quantile based measures
    norm_q_1, norm_q_2 = _bf_nomalized_quantile_measures(q_1, q_2)

    return q_1, q_2, norm_q_1, norm_q_2


def _bf_unconditional_sample(
    n_draws,
    n_params,
    dist_type,
    loc,
    scale,
    sampling_scheme="sobol",
    seed=0,
    skip=0,
):
    """Generate two independent sample sets."""
    # Generate uniform distributed sample
    np.random.seed(seed)
    if sampling_scheme == "sobol":
        u = cp.generate_samples(
            order=n_draws + skip,
            domain=n_params,
            rule="S",
        ).T
    elif sampling_scheme == "random":
        u = np.random.uniform(size=(n_draws, n_params))
    else:
        raise ValueError("Argument 'sampling_scheme' is not in {'sobol', 'random'}.")

    skip = skip if sampling_scheme == "sobol" else 0

    u = cp.generate_samples(order=n_draws, domain=2 * n_params, rule="S").T
    u_1 = u[skip:, :n_params]
    u_2 = u[skip:, n_params:]

    # Transform uniform draw into the assigned joint PDF
    if dist_type == "Normal":
        z = norm.ppf(u_1)
        z_prime = norm.ppf(u_2)
        cholesky = np.linalg.cholesky(scale)
        x = loc + cholesky.dot(z.T).T
        x_prime = loc + cholesky.dot(z_prime.T).T
    elif dist_type == "Exponential":
        x = expon.ppf(u_1, loc, scale)
        x_prime = expon.ppf(u_2, loc, scale)
    elif dist_type == "Uniform":
        x = uniform.ppf(u_1, loc, scale)
        x_prime = uniform.ppf(u_2, loc, scale)
    else:
        raise NotImplementedError

    return x, x_prime


@jit(nopython=True)
def _bf_conditional_sample(x, x_prime):
    """Generate a mixed sample set distributed accroding to the conditional PDF."""
    n_draws, n_params = x.shape
    x_mix = np.zeros((n_draws, n_params, n_draws, n_params))

    for i in range(n_params):
        for j in range(n_draws):
            x_mix[j, i] = x
            x_mix[j, i, :, i] = x_prime[j, i]

    return x_mix


def _bf_unconditional_quantile_y(x, alp, func):
    """Calculate quantiles of outputs with unconditional sample set as input."""
    n_draws, n_params = x.shape

    y_x = np.zeros((n_draws, n_draws, 1))  # N*N*1
    y_x_asc = np.zeros((n_draws, n_draws, 1))  # N*N*1
    quantile_y_x = np.zeros((n_draws, len(alp), 1))  # N*31*1

    for j in range(n_draws):
        y_x[j] = np.vstack(func(x))
        y_x_asc[j] = np.sort(y_x[j], axis=0)
        # conditioanl q_y(alp)
        for pp in range(len(alp)):
            quantile_y_x[j, pp] = y_x_asc[j][
                (np.floor(alp[pp] * n_draws)).astype(int)
            ]  # quantiles corresponding to alpha

    return quantile_y_x


def _bf_conditional_quantile_y(x_mix, alp, func):
    """Calculate quantiles of outputs with conditional sample set as input."""
    n_draws, n_params = x_mix.shape[:2]

    y_x_mix = np.zeros((n_draws, n_params, n_draws, 1))  # N*d*N*1
    y_x_mix_asc = np.zeros((n_draws, n_params, n_draws, 1))  # N*d*N*1
    quantile_y_x_mix = np.zeros((n_draws, n_params, len(alp), 1))  # N*4*31*1

    # Equation 21(b) & 23
    for j in range(n_draws):
        for i in range(n_params):
            # values of conditional outputs
            y_x_mix[j, i] = np.vstack(func(x_mix[j, i]))
            y_x_mix_asc[j, i] = np.sort(y_x_mix[j, i], axis=0)
            # conditioanl q_y(alp)
            for pp in range(len(alp)):
                quantile_y_x_mix[j, i, pp] = y_x_mix_asc[j, i][
                    (np.floor(alp[pp] * n_draws)).astype(int)
                ]  # quantiles corresponding to alpha

    return quantile_y_x_mix


@jit(nopython=True)
def _bf_quantile_measures(
    quantile_y_x,
    quantile_y_x_mix,
):
    """Compute the brute force MC/QMC estimators of quantile based measures."""
    # initialization
    n_draws, n_params, len_alp = quantile_y_x_mix.shape[:3]

    q_1 = np.zeros((n_params, len_alp, 1))  # d*31*1
    q_2 = np.zeros((n_params, len_alp, 1))  # d*31*1
    delt = np.zeros((n_draws, n_params, len_alp, 1))  # N*d*31*1

    # Equation 24 & 25
    for j in range(n_draws):
        delt[j] = quantile_y_x_mix[j] - quantile_y_x[j]  # delt
        for i in range(n_params):
            for pp in range(len_alp):
                q_1[i, pp] = np.mean(np.absolute(delt[:, i, pp]))  # |delt|
                q_2[i, pp] = np.mean(delt[:, i, pp] ** 2)  # (delt)^2

    # reshape
    q_1 = np.transpose(q_1).reshape((len_alp, n_params))
    q_2 = np.transpose(q_2).reshape((len_alp, n_params))

    return q_1, q_2


def _bf_nomalized_quantile_measures(q_1, q_2):
    """Compute the brute force MC/QMC estimators of nomalized quantile based measures."""
    len_alp, n_params = q_1.shape

    sum_q_1 = np.zeros(len_alp)
    sum_q_2 = np.zeros(len_alp)
    norm_q_1 = np.zeros((len_alp, n_params))
    norm_q_2 = np.zeros((len_alp, n_params))

    # Equation 13 & 14
    for pp in range(len_alp):
        sum_q_1[pp] = np.sum(q_1[pp, :])
        sum_q_2[pp] = np.sum(q_2[pp, :])
        for i in range(n_params):
            norm_q_1[pp, i] = q_1[pp, i] / sum_q_1[pp]
            norm_q_2[pp, i] = q_2[pp, i] / sum_q_2[pp]

    return norm_q_1, norm_q_2
