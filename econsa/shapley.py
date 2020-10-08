"""Capabilities for computation of Shapley effects.

This module contains functions to estimate shapley effects for models with
dependent inputs.

"""
import itertools

import chaospy as cp
import numpy as np
import pandas as pd

from econsa.sampling import cond_mvn


def get_shapley(
    method,
    model,
    x_all,
    x_cond,
    n_perms,
    n_inputs,
    n_output,
    n_outer,
    n_inner,
):
    """Shapley value function.

    This function calculates Shapley effects and their standard errors for
    models with both dependent and independent inputs. We allow for two ways
    to calculate Shapley effects: by examining all permutations of the given
    inputs or alternatively, by randomly sampling permutations of inputs.

    The function is a translation of the exact and random permutation funtions
    in the ``sensitivity`` package in R, and takes the method (exact or random)
    as an argument and therefore estimates shapley effects in both ways.


    Parameters
    ----------
    method : string
           Specifies which method you want to use to estimate shapley effects,
           the ``exact`` or ``random`` permutations method. When the number of
           inputs is small, it is better to use the ``exact`` method, and
           ``random`` otherwise.

    model : string
          The model/function you will calculate the shapley effects on.

    x_all : string (n)
         A function that takes `n´ as an argument and generates a n-sample of
         a d-dimensional input vector.

    x_cond: string (n, Sj, Sjc, xjc)
         A function that takes `n, Sj, Sjc, xjc´ as arguments and generates
         a n- sample an input vector corresponding to the indices in `Sj´
         conditional on the input values `xjc´ with the index set `Sjc´.

    n_perms : scalar
            This is an input for the number of permutations you want the model
            to make. For the ``exact`` method, this argument is none as the
            number of permutations is determined by how many inputs you have,
            and for the ``random`` method, this is determined exogeniously.

    n_inputs : scalar
             The number of input vectors for which shapley estimates are being
             estimated.

    n_output : scalar
             Monte Carlo (MC) sample size to estimate the output variance of
             the model output `Y´.

    n_outer : scalar
            The outer Monte Carlo sample size to estimate the cost function for
            `c(J) = E[Var[Y|X]]´.

    n_inner : scalar
            The inner Monte Carlo sample size to estimate the cost function for
            `c(J) = Var[Y|X]´.

    Returns
    -------
    effects : DataFrame
            n dimensional DataFrame with the estimated shapley effects, the
            standard errors and the confidence intervals for the input vectors.

    Sources
    -------
    R's `sensitiity´ package for the shapleyPermEx_ and shapleyPermRand_ functions.

    .. _shapleyPermEx: https://rdrr.io/cran/sensitivity/src/R/shapleyPermEx.R

    .. _shapleyPermRand: https://rdrr.io/cran/sensitivity/src/R/shapleyPermRand.R

    Contributor
    -----------
    Linda Maokomatanda

    """
    if method == "exact":
        permutations = list(itertools.permutations(range(n_inputs), n_inputs))
        permutations = [list(i) for i in permutations]

        n_perms = len(permutations)
    else:
        permutations = np.zeros((n_perms, n_inputs), dtype=np.int64)
        for i in range(n_perms):
            permutations[i] = np.random.permutation(n_inputs)

        n_perms = np.int(permutations.shape[0])

    # initiate empty input array for sampling
    model_inputs = np.zeros(
        (n_output + n_perms * (n_inputs - 1) * n_outer * n_inner, n_inputs),
    )
    model_inputs[:n_output, :] = x_all(n_output).T

    for p in range(n_perms):

        perms = permutations[p]
        perms_sorted = np.argsort(perms)

        for j in range(1, n_inputs):
            # set of the 0st-(j-1)th elements in perms
            sj = perms[:j]
            # set of the jth-n_perms elements in perms
            sjc = perms[j:]

            # sampled values of the inputs in Sjc
            xjc_sampled = np.array(x_cond(n_outer, sjc, None, None)).T

            for length in range(n_outer):
                xjc = xjc_sampled[
                    length,
                ]

                # sample values of inputs in Sj conditional on xjc
                sample_inputs = np.array(x_cond(n_inner, sj, sjc, xjc.flat)).T.reshape(
                    n_inner,
                    -1,
                )
                concatenated_sample = np.concatenate(
                    (sample_inputs, np.ones((n_inner, 1)) * xjc),
                    axis=1,
                )
                inner_indices = (
                    n_output
                    + p * (n_inputs - 1) * n_outer * n_inner
                    + (j - 1) * n_outer * n_inner
                    + length * n_inner
                )
                model_inputs[
                    inner_indices : (inner_indices + n_inner), :
                ] = concatenated_sample[:, perms_sorted]

    # calculate model output
    output = model(model_inputs)

    # Initialize Shapley, main and total Sobol effects for all players
    shapley_effects = np.zeros(n_inputs)
    shapley_effects_squared = np.zeros(n_inputs)

    # estimate the variance of the model output
    model_output = output[:n_output]
    output = output[n_output:]
    output_variance = np.var(model_output)

    # estimate shapley, main and total sobol effects
    conditional_variance = np.zeros(n_outer)

    for p in range(n_perms):

        perms = permutations[p]
        previous_cost = 0

        for j in range(n_inputs):
            if j == (n_inputs - 1):
                estimated_cost = output_variance
                delta = estimated_cost - previous_cost

            else:
                for length in range(n_outer):
                    model_output = output[:n_inner]
                    output = output[n_inner:]
                    conditional_variance[length] = np.var(model_output, ddof=1)
                estimated_cost = np.mean(conditional_variance)
                delta = estimated_cost - previous_cost

            shapley_effects[perms[j]] = shapley_effects[perms[j]] + delta
            shapley_effects_squared[perms[j]] = (
                shapley_effects_squared[perms[j]] + delta ** 2
            )

            previous_cost = estimated_cost

    shapley_effects = shapley_effects / n_perms / output_variance
    shapley_effects_squared = shapley_effects_squared / n_perms / (output_variance ** 2)
    standard_errors = np.sqrt(
        (shapley_effects_squared - shapley_effects ** 2) / n_perms,
    )

    # confidence intervals
    ci_min = shapley_effects - 1.96 * standard_errors
    ci_max = shapley_effects + 1.96 * standard_errors

    col = ["X" + str(i) for i in np.arange(n_inputs) + 1]

    effects = pd.DataFrame(
        np.array([shapley_effects, standard_errors, ci_min, ci_max]),
        index=["Shapley effects", "std. errors", "CI_min", "CI_max"],
        columns=col,
    ).T

    return effects


def _r_condmvn(
    n,
    mean,
    cov,
    dependent_ind,
    given_ind,
    x_given,
):
    """Function to generate conditional law.

    Function to simulate conditional gaussian distribution of X[dependent.ind]
    | X[given.ind] = X.given where X is multivariateNormal(mean = mean, covariance = cov)

    """
    cond_mean, cond_var = cond_mvn(
        mean,
        cov,
        dependent_ind=dependent_ind,
        given_ind=given_ind,
        given_value=x_given,
    )
    distribution = cp.MvNormal(cond_mean, cond_var)

    return distribution.sample(n)
