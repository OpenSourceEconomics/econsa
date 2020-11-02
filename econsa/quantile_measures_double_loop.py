"""The double loop reordering estimators of quantile based global sensitivity measures.

This module contains functions to calculate global sensitivity measures based on
quantiles of the output introduced by Kucherenko et al.(2019).

"""
import chaospy as cp
import numpy as np
from numba import jit
from scipy.stats import expon
from scipy.stats import norm
from scipy.stats import uniform


def dlr_mcs_quantile(
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
    (DLR) approach described in Section 4.2 of [K2019]_.

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

    # get the base sample from a joint PDF
    x = _dlr_unconditional_sample(
        n_params,
        loc,
        scale,
        dist_type,
        n_draws,
        sampling_scheme="sobol",
        seed=0,
        skip=0,
    )

    # get the conditional sample set
    x_mix = _dlr_conditional_sample(x)

    # quantile of output calculated with base sample set
    quantile_y_x = _dlr_unconditional_quantile_y(
        x,
        func,
        alp,
    )

    # quantile of output calculated with conditional sample set
    quantile_y_x_mix = _dlr_conditional_quantile_y(x_mix, func, alp)

    # Get quantile based measures
    q_1, q_2 = _dlr_quantile_measures(quantile_y_x, quantile_y_x_mix)

    # Get normalized quantile based measures
    norm_q_1, norm_q_2 = _dlr_normalized_quantile_measures(q_1, q_2)

    return q_1, q_2, norm_q_1, norm_q_2


def _dlr_unconditional_sample(
    n_params,
    loc,
    scale,
    dist_type,
    n_draws,
    sampling_scheme="sobol",
    seed=0,
    skip=0,
):
    """Generate the base sample set according to a joint PDF."""
    # Generate uniform distributed sample
    np.random.seed(seed)

    if sampling_scheme == "sobol":
        u_1 = cp.generate_samples(
            order=n_draws + skip,
            domain=n_params,
            rule="S",
        ).T
    elif sampling_scheme == "random":
        u_1 = np.random.uniform(size=(n_draws, n_params))
    else:
        raise ValueError("Argument 'sampling_scheme' is not in {'sobol', 'random'}.")

    skip = skip if sampling_scheme == "sobol" else 0

    u_2 = u_1[skip:, :n_params]

    # Transform uniform draw into the assigned joint PDF
    if dist_type == "Normal":
        z = norm.ppf(u_2)
        cholesky = np.linalg.cholesky(scale)
        x = loc + cholesky.dot(z.T).T
    elif dist_type == "Exponential":
        x = expon.ppf(u_2, loc, scale)
    elif dist_type == "Uniform":
        x = uniform.ppf(u_2, loc, scale)
    else:
        raise NotImplementedError

    return x


@jit(nopython=True)
def _dlr_conditional_sample(x):
    """Generate a conditional sample set from the base sample set."""

    n_draws, n_params = x.shape

    # The dependence of m versus n_draws accroding to [K2017] fig.1
    if n_draws == 2 ** 6:
        m = 2 ** 3
    elif n_draws <= 2 ** 9:
        m = 2 ** 4
    elif n_draws == 2 ** 10:
        m = 2 ** 5
    elif n_draws <= 2 ** 13:
        m = 2 ** 6
    elif n_draws <= 2 ** 15:
        m = 2 ** 7
    else:
        raise NotImplementedError

    conditional_bin = x[:m]
    x_mix = np.zeros((m, n_params, n_draws, n_params))

    # subdivide unconditional sample into M eaually bins, within each bin x_i being fixed.
    for i in range(n_params):
        for j in range(m):
            x_mix[j, i] = x
            x_mix[j, i, :, i] = conditional_bin[j, i]

    return x_mix


def _dlr_unconditional_quantile_y(
    x,
    func,
    alp,
):
    """Calculate quantiles of outputs with unconditional sample set as input."""
    n_draws = len(x)
    # Equation 26 & 23
    y_x = func(x)  # values of outputs
    y_x_asc = np.sort(y_x)  # reorder in ascending order
    q_index = (np.floor(alp * n_draws)).astype(int)
    quantile_y_x = y_x_asc[q_index]  # quantiles corresponding to alpha

    return quantile_y_x


def _dlr_conditional_quantile_y(x_mix, func, alp):
    """Calculate quantiles of outputs with conditional sample set as input."""
    m, n_params, n_draws = x_mix.shape[:3]

    y_x_mix = np.zeros((m, n_params, n_draws, 1))  # N*d*N*1
    y_x_mix_asc = np.zeros((m, n_params, n_draws, 1))  # N*d*N*1
    quantile_y_x_mix = np.zeros((1, n_params, len(alp), m))  # 1*d*31*m

    # Equation 26 & 23. Get CDF within each bin.
    for i in range(n_params):
        for j in range(m):
            # values of conditional outputs
            y_x_mix[j, i] = np.vstack(func(x_mix[j, i]))
            y_x_mix[j, i].sort(axis=0)
            y_x_mix_asc[j, i] = y_x_mix[j, i]
            for pp in range(len(alp)):
                quantile_y_x_mix[0, i, pp, j] = y_x_mix_asc[j, i][
                    (np.floor(alp[pp] * n_draws)).astype(int)
                ]  # quantiles corresponding to alpha
    return quantile_y_x_mix


@jit(nopython=True)
def _dlr_quantile_measures(quantile_y_x, quantile_y_x_mix):
    """Compute DLR MC/QMC estimators of quantile based measures."""
    n_params, len_alp, m = quantile_y_x_mix.shape[1:]

    # initialization
    q_1 = np.zeros((len_alp, n_params))
    q_2 = np.zeros((len_alp, n_params))
    delt = np.zeros((1, n_params, 1, m))

    # Equation 27 & 28
    for i in range(n_params):
        for pp in range(len_alp):
            delt[0, i] = quantile_y_x_mix[0, i, pp, :] - quantile_y_x[pp]  # delt
            q_1[pp, i] = np.mean(np.absolute(delt[0, i]))  # |delt|
            q_2[pp, i] = np.mean(delt[0, i] ** 2)  # (delt)^2

    return q_1, q_2


def _dlr_normalized_quantile_measures(q_1, q_2):
    """Compute DLR MC/QMC estimators of normalized quantile based measures."""
    len_alp, n_params = q_2.shape

    # initialize quantile measures arrays.
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
