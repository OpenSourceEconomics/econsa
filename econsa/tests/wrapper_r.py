"""Wrapping R.

This module contains all functionality related to the use of functions from R for testing purposes.

"""
import numpy as np
import rpy2.robjects.packages as rpackages
from rpy2 import robjects
from rpy2.robjects import numpy2ri

r_package_cond_mvnorm = rpackages.importr("condMVNorm")


def r_cond_mvn(mean, cov, dependent_ind, given_ind, given_value):
    """The original function for `cond_mvn`."""
    numpy2ri.activate()
    r_mean = robjects.FloatVector(mean)
    n = cov.shape[0]
    r_cov = robjects.r.matrix(cov, n, n)
    r_dependent_ind = robjects.IntVector([x + 1 for x in dependent_ind])
    r_given_ind = robjects.IntVector([x + 1 for x in given_ind])
    r_given_value = robjects.FloatVector(given_value)

    args = (r_mean, r_cov, r_dependent_ind, r_given_ind, r_given_value)
    r_cond_mean, r_cond_cov = r_package_cond_mvnorm.condMVN(*args)

    r_cond_mean, r_cond_cov = np.array(r_cond_mean), np.array(r_cond_cov)

    numpy2ri.deactivate()

    return r_cond_mean, r_cond_cov
