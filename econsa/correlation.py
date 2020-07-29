"""Map arbitrary correlation matrix to Gaussian.

This module implements methods from two papers to map arbitrary correlation
matrix into correlation matrix for Gaussian copulas.
"""
from functools import partial

import chaospy as cp
import numpy as np
from statsmodels.stats.correlation_tools import corr_nearest


def gc_correlation(marginals, corr, force_calc=False, rule="sobol", num_points=10000):
    """Correlation for Gaussian copula.

    This function implements the algorithm outlined in Section 4.2 of [K2012]_
    to map arbitrary correlation matrix to an correlation matrix for
    Gaussian copula.
    For special combination of distributions, use the values from Table 4. of
    [L1986]_.

    Since chaospy's copula functions only accept positive definite correlation matrix,
    this function also checks the output,
    and transforms to nearest positive definite matrix if it is not already.

    Parameters
    ----------
    marginals : chaospy.distributions
        Marginal distributions of the correlated variables. All marginals must be chaospy
        distributions, otherwise returns error.

    corr : array_like
        The correlation matrix to be transformed.

    force_calc : bool, optional
        When `True`, calculate the covariances ignoring all special combinations of marginals
        (default value is `False`).

    Returns
    -------
    gc_corr : numpy.ndarray
        The transformed correlation matrix that is ready to be fed into a Gaussian copula.

    References
    ----------
    .. [K2012] Kucherenko, S., Tarantola, S., & Annoni, P. (2012).
        Estimation of global sensitivity indices for models with
        dependent variables. Computer Physics Communications, 183(4), 937–946.

    .. [L1986] Liu, P.-L., & Der Kiureghian, A. (1986).
        Multivariate distribution models with prescribed marginals
        and covariances. Probabilistic Engineering Mechanics, 1(2), 105–112.

    Examples
    --------
    >>> corr = [[1.0, 0.6, 0.3], [0.6, 1.0, 0.0], [0.3, 0.0, 1.0]]
    >>> marginals = [cp.Normal(1.00), cp.Uniform(lower=-4.00), cp.Normal(4.20)]
    >>> corr_transformed = gc_correlation(marginals, corr)
    >>> copula = cp.Nataf(cp.J(*marginals), corr_transformed)
    >>> corr_copula = np.corrcoef(copula.sample(1000000))
    >>> np.testing.assert_almost_equal(corr, corr_copula, decimal=6)
    """
    corr = np.atleast_2d(corr)

    # Test that marginals are all cp.distributions.
    for marginal in marginals:
        if issubclass(type(marginal), cp.distributions.baseclass.Dist):
            continue
        else:
            raise NotImplementedError("marginals must be chaospy distributions")

    # check_corr
    if not np.allclose(corr, corr.T):
        raise ValueError("corr is not a symmetric matrix")
    elif not np.all((corr >= -1) & (corr <= 1)):
        raise ValueError("corr must be between 0 and 1")
    elif not np.all(np.diagonal(corr) == 1):
        raise ValueError("the diagonal of corr must all be 1")
    elif np.all(np.linalg.eigvals(corr) > 0) == 0:
        raise ValueError("corr is not positive-definite")

    dim = len(corr)
    indices = np.tril_indices(dim, -1)
    gc_corr = np.identity(dim)

    for i, j in list(zip(*indices)):
        subset = [marginals[i], marginals[j]]
        distributions, rho = cp.J(*subset), corr[i, j]
        gc_corr[i, j] = _gc_correlation_pairwise(
            distributions, rho, force_calc, num_points, rule,
        )

    # Align upper triangular with lower triangular.
    gc_corr = gc_corr + gc_corr.T - np.diag(np.diag(gc_corr))

    # If gc_corr is not positive definite, find the nearest one.
    gc_corr = _find_positive_definite(gc_corr)

    return gc_corr


def _gc_correlation_pairwise(
    distributions, rho, force_calc, num_points=1000, rule="sobol",
):
    assert len(distributions) == 2

    if force_calc and type(_special_dist(distributions)) is not bool:
        result = rho * _special_dist(distributions)[1]
    else:

        arg_1 = np.prod(cp.E(distributions))
        arg_2 = np.sqrt(np.prod(cp.Var(distributions)))
        arg = rho * arg_2 + arg_1

        kwargs = dict()
        kwargs["distributions"] = distributions
        kwargs["num_points"] = num_points
        kwargs["rule"] = rule
        kwargs["arg"] = arg

        grid = np.linspace(-0.99, 0.99, num=199, endpoint=True)
        v_p_criterion = np.vectorize(partial(_criterion, **kwargs))
        result = grid[np.argmin(v_p_criterion(grid))]

    return result


def _find_positive_definite(cov):
    """Find the nearest positive definite matrix."""
    if np.all(np.linalg.eigvalsh(cov) > 0) == 0:
        while True:
            cov_new = corr_nearest(cov)
            if np.all(np.linalg.eigvalsh(cov_new) > 0) == 1:
                cov = cov_new
                break
    return cov


def _special_dist(distributions):
    """Test whether [L1986]_ is applicable."""
    dist_type = list()
    for dist in distributions:
        dist_type.append(str(dist).split(sep="(", maxsplit=1)[0].lower())

    success = True

    if any(dist_type) == "normal":
        dist_norm = dist_type.index("normal")
        dist_type.pop(dist_norm)
        dist_other = dist_type[0]
        if "normal" == dist_other:
            f = 1
        elif "uniform" == dist_other:
            f = 1.023
        elif "exponential" == dist_other:
            f = 1.107
        elif "rayleigh" == dist_other:
            f = 1.014
        elif "logweibull" == dist_other:
            # Type-I extreme value, Gumbel
            f = 1.031
        else:
            success = False
    else:
        success = False

    if success:
        return success, f
    else:
        return success


def _criterion(rho_c, arg, distributions, num_points, rule):
    cov = np.identity(2)
    cov[1, 0] = cov[0, 1] = rho_c
    distribution = cp.MvNormal(loc=np.zeros(2), scale=cov)
    draws = distribution.sample(num_points, rule=rule).T.reshape(num_points, 2)
    x_1, x_2 = np.split(draws, 2, axis=1)

    standard_norm_cdf = cp.Normal().cdf
    arg_1 = distributions[0].inv(standard_norm_cdf(x_1))
    arg_2 = distributions[1].inv(standard_norm_cdf(x_2))
    point = arg_1 * arg_2

    return (np.mean(point) - arg) ** 2
