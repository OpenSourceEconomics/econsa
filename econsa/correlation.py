"""Map arbitrary correlation matrix to Gaussian.

This module implements methods from two papers to map arbitrary correlation
matrix into correlation matrix for Gaussian copulas.
"""
import chaospy as cp
import numpy as np
from scipy import optimize
from scipy.stats import multivariate_normal as multivariate_norm


def gc_correlation(marginals, corr, check_corr=True):
    """Correlation for Gaussian copula.

    This function implements the algorithm outlined in Section 4.2 of [K2012]_
    to map arbitrary correlation matrix to an correlation matrix for
    Gaussian copula.
    For special combination of distributions, use the values from Table 4. of
    [L1986]_.

    Parameters
    ----------
    marginals : chaospy.distributions
        Marginal distributions of the correlated variables. All marginals must be chaospy
        distributions, otherwise returns error.

    corr : array_like
        The correlation matrix to be transformed.

    check_corr : bool, optional
        Check that `corr` is symmetric, all elements are beteween 0 and 1,
        all diagonal elements are 1,
        and all eigenvalue is positive (default value is `True`).

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
    >>> np.testing.assert_almost_equal(corr, corr_copula, decimal=1)
    """
    corr = np.atleast_2d(corr)

    # Test that marginals are all cp.distributions.
    for marginal in marginals:
        if issubclass(type(marginal), cp.distributions.baseclass.Dist):
            continue
        else:
            raise NotImplementedError("marginals must be chaospy distributions")

    if check_corr:
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
        gc_corr[i, j] = _gc_correlation_pairwise(distributions, rho)

    # Align upper triangular with lower triangular.
    gc_corr = gc_corr + gc_corr.T - np.diag(np.diag(gc_corr))

    return gc_corr


def _gc_correlation_pairwise(distributions, rho, seed=123, num_draws=100000):

    assert len(distributions) == 2

    # Test whether [L1986]_ is applicable
    # Extract types of distributions
    dist_type = list()
    for dist in distributions:
        dist_type.append(str(dist).split(sep="(", maxsplit=1)[0].lower())

    try:
        dist_norm = dist_type.index("normal")
        dist_type.pop(dist_norm)
        dist_other = dist_type[0]

        # TODO: the code here only includes table 4,
        # we want to also also incorporate table 5.
        if "uniform" == dist_other:
            f = 1.023
        elif "exponential" == dist_other:
            f = 1.107
        elif "rayleigh" == dist_other:
            f = 1.014
        elif "logweibull" == dist_other:
            # Type-I extreme value, Gumbel
            f = 1.031
        else:
            raise ValueError("This combination is not implemented.")

        result = rho * f
    except ValueError:
        arg_1 = np.prod(cp.E(distributions))
        arg_2 = np.sqrt(np.prod(cp.Var(distributions)))
        arg = rho * arg_2 + arg_1

        kwargs = dict()
        kwargs["args"] = (arg, distributions, seed, num_draws)
        kwargs["bounds"] = (-0.99, 0.99)
        kwargs["method"] = "bounded"

        out = optimize.minimize_scalar(_criterion, **kwargs)
        assert out["success"]
        result = out["x"]

    return result


def _criterion(rho_c, arg, distributions, seed, num_draws):

    cov = np.identity(2)
    cov[1, 0] = cov[0, 1] = rho_c

    np.random.seed(seed)

    # TODO: Here we need to use proper quadrature rules
    # instead of Monte Carlo integration.
    x_1, x_2 = multivariate_norm([0, 0], cov).rvs(num_draws).T

    standard_norm_cdf = cp.Normal().cdf
    arg_1 = distributions[0].inv(standard_norm_cdf(x_1))
    arg_2 = distributions[1].inv(standard_norm_cdf(x_2))
    point = arg_1 * arg_2

    return (np.mean(point) - arg) ** 2