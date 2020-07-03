"""Capabilities for sampling of random input parameters.

This module contains functions to sample random input parameters.

"""
import numpy as np


def cond_mvn(
    mean, sigma, dependent_ind, given_ind=None, given_value=None, check_sigma=True,
):
    r"""Conditional mean and variance function.

    This function provides the conditional mean and variance-covariance matrix of [Y given X],
    where Z = (X,Y) is the fully-joint multivariate normal distribution with mean equal to
    ``mean`` and covariance matrix ``sigma``.

    This is a translation of the main function of R package condMVNorm_.

    .. _condMVNorm: https://cran.r-project.org/package=condMVNorm


    Parameters
    ----------
    mean : array_like
           The mean vector of the multivariate normal distribution.

    sigma : array_like
            Symmetric and positive-definite covariance matrix of
            the multivariate normal distribution.

    dependent_ind : array_like
                    The indices of dependent variables.

    given_ind : array_like, optional
                The indices of independent variables (default value is ``None``).
                If not specified return unconditional values.

    given_value : array_like, optional
              The conditioning values (default value is ``None``). Should be the same length as
              ``given_ind``, otherwise throw an error.

    check_sigma : bool, optional
                  Check that ``sigma`` is symmetric, and no eigenvalue is zero (default value is
                  ``True``).

    Returns
    -------
    cond_mean : array_like
                The conditional mean of dependent variables.

    cond_cov : arrray_like
               The conditional covariance matrix of dependent variables.

    Examples
    --------
    >>> mean = np.array([1, 1, 1])
    >>> sigma = np.array([[4.0677098, -0.9620331, 0.9897267],
    ...                   [-0.9620331, 2.2775449, 0.7475968],
    ...                   [0.9897267, 0.7475968, 0.7336631]])
    >>> dependent_ind = [0, ]
    >>> given_ind = [1, 2]
    >>> given_value = [1, -1]
    >>> cond_mean, cond_cov = cond_mvn(mean, sigma, dependent_ind, given_ind, given_value)
    >>> np.testing.assert_almost_equal(cond_mean, -4.347531, decimal=6)
    >>> np.testing.assert_almost_equal(cond_cov, 0.170718, decimal=6)
    """
    mean_np = np.array(mean)
    sigma_np = np.array(sigma)
    given_ind_np = np.array(given_ind, ndmin=1)
    given_value_np = np.array(given_value, ndmin=1)

    # Check `sigma` is symmetric and positive-definite:
    if check_sigma:
        if not np.allclose(sigma_np, sigma_np.T):
            raise ValueError("sigma is not a symmetric matrix")
        elif np.all(np.linalg.eigvals(sigma_np) > 0) == 0:
            raise ValueError("sigma is not positive-definite")

    # When `given_ind` is None, return mean and variances of dependent values:
    if given_ind is None:
        cond_mean = np.array(mean_np[dependent_ind])
        cond_cov = np.array(sigma_np[dependent_ind, :][:, dependent_ind])
        return cond_mean, cond_cov

    # Make sure that `given_value` aligns with `given_len`. This includes the case that
    # `given_value` is empty.
    if len(given_value_np) != len(given_ind_np):
        raise ValueError("lengths of given_value and given_ind must be the same")

    b = sigma_np[dependent_ind, :][:, dependent_ind]
    c = sigma_np[dependent_ind, :][:, given_ind]
    d = sigma_np[given_ind, :][:, given_ind]
    c_dinv = c @ np.linalg.inv(d)

    cond_mean = mean_np[dependent_ind] + c_dinv @ (given_value - mean_np[given_ind])
    cond_cov = b - c_dinv @ (c.T)

    return cond_mean, cond_cov
