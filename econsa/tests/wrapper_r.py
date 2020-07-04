"""Wrapping R.

This module contains all functionality related to the use of functions from R for testing purposes.

"""
import numpy as np
import rpy2.robjects.packages as rpackages
from rpy2 import robjects
from rpy2.robjects import numpy2ri

r_package_cond_mvnorm = rpackages.importr("condMVNorm")


def r_cond_mvn(mean, sigma, dependent_ind, given_ind, given_value):
    numpy2ri.activate()
    r_mean = robjects.FloatVector(mean)
    n = sigma.shape[0]
    r_sigma = robjects.r.matrix(sigma, n, n)
    r_dependent_ind = robjects.IntVector([x + 1 for x in dependent_ind])
    r_given_ind = robjects.IntVector([x + 1 for x in given_ind])
    r_given_value = robjects.IntVector(given_value)

    args = (r_mean, r_sigma, r_dependent_ind, r_given_ind, r_given_value)
    r_cond_mean, r_cond_cov = r_package_cond_mvnorm.condMVN(*args)

    r_cond_mean, r_cond_cov = np.array(r_cond_mean), np.array(r_cond_cov)

    numpy2ri.deactivate()

    return r_cond_mean, r_cond_cov
