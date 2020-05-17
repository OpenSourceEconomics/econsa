# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# import numba as nb
import numpy as np
import pandas as pd
import scipy.stats as stats
import chaospy as cp
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns

from econsa.morris import (
    _shift_cov,
    _shift_sample,
    _uniform_to_standard_normal,
    elementary_effects,
)

# # Replicating: [Introducing Copula in Monte Carlo Simulation](https://towardsdatascience.com/introducing-copula-in-monte-carlo-simulation-9ed1fe9f905)

# ## Random Variable Transformation

# +
# Generate Monte Carlo sample

x = stats.uniform(0, 1).rvs(10000)
# -

plt.clf()
sns.distplot(x)

norm = stats.distributions.norm()
x_trans = norm.ppf(x)

plt.clf()
sns.distplot(x_trans)

plt.clf()
sns.jointplot(x=x, y=x_trans)

# ## Gaussian Copula — Adding Variable Correlations

mvnorm = stats.multivariate_normal([0, 0], [[1., 0.5], [0.5, 1.]])
x = mvnorm.rvs((10000,))

sns.jointplot(x=x[:,0], y=x[:,1], kind="kde")

norm = stats.norm([0],[1])
x_unif = norm.cdf(x)

plt.clf()
sns.jointplot(x=x_unif[:,0], y=x_unif[:,1], kind="hex")

x1_tri  = stats.triang.ppf(x_unif[:, 0],  c=0.158 , loc=36, scale=21)
x2_norm =stats.norm(525, 112).ppf(x_unif[:, 1])

plt.clf()
sns.distplot(x1_tri)

plt.clf()
sns.distplot(x2_norm)

plt.clf()
sns.jointplot(x=x1_tri, y=x2_norm, kind="hex")

# ## Probabilistic Estimation of HCIIP

# ### No Variable Correlation Case

# +
# HCIIP = GRV*NTG*POR*SHC/FVF

means = [0.]*5
cov = [[1., 0., 0., 0., 0.],
[0., 1., 0., 0., 0.],
[0., 0., 1., 0., 0.],
[0., 0., 0., 1., 0.],
[0., 0., 0., 0., 1.]]

mvnorm_std = stats.multivariate_normal(means,cov)
x = mvnorm_std.rvs(10000,random_state=42)
norm_std = stats.norm()
x_unif = norm_std.cdf(x)

#create individual distr.
grv = stats.triang(c=0.1 , loc=10000, scale=300).ppf(x_unif[:, 0])
ntg = stats.triang(c=0.2 , loc=0.5, scale=0.5).ppf(x_unif[:, 1])
phi = stats.truncnorm(-2*1.96,1.96,0.2,0.05).ppf(x_unif[:, 2])
shc = stats.norm(0.6,0.05).ppf(x_unif[:, 3])
fvf= stats.truncnorm(-1.96,2*1.96,1.3,0.1).ppf(x_unif[:, 4])

stoiip = 7758*grv*ntg*phi*shc/fvf/1e6
# -

plt.clf()
sns.distplot(stoiip, kde=False)

plt.clf()
sns.distplot(stoiip, hist_kws=dict(cumulative=True))

# ### Variable Correlation Case

# +
means = [0.]*5

cov = [[1., 0., 0., 0., 0.],
[0., 1., 0.7, 0.6, 0.],
[0., 0.7, 1., 0.8, 0.],
[0., 0.6, 0.8, 1., 0.],
[0., 0., 0., 0., 1.]]

mvnorm_std = stats.multivariate_normal(means,cov)
x = mvnorm_std.rvs(10000,random_state=42)
norm_std = stats.norm()
x_unif = norm_std.cdf(x)

#create individual distr.
grv = stats.triang(c=0.1 , loc=10000, scale=300).ppf(x_unif[:, 0])
ntg = stats.triang(c=0.2 , loc=0.5, scale=0.5).ppf(x_unif[:, 1])
phi = stats.truncnorm(-2*1.96,1.96,0.2,0.05).ppf(x_unif[:, 2])
shc = stats.norm(0.6,0.05).ppf(x_unif[:, 3])
fvf= stats.truncnorm(-1.96,2*1.96,1.3,0.1).ppf(x_unif[:, 4])

stoiip = 7758*grv*ntg*phi*shc/fvf/1e6
# -

plt.clf()
sns.distplot(stoiip, kde=False)

plt.clf()
sns.distplot(stoiip, hist_kws=dict(cumulative=True))

# # Testing ChaosPy: [Distributions — ChaosPy documentation](https://chaospy.readthedocs.io/en/master/distributions/index.html)

# to create a Gaussian random variable:
distribution = cp.Normal(mu=2, sigma=2)

# to create values from the probability density function:
t = np.linspace(-3, 3, 9)
distribution.pdf(t).round(3)

# create values from the cumulative distribution function:
distribution.cdf(t).round(3)

# To be able to perform any Monte Carlo method,
# each distribution contains random number generator:
distribution.sample(6).round(4)

plt.clf()
sns.distplot(distribution.pdf(t).round(3), kde=False)

# to create low-discrepancy Hammersley sequences
# samples combined with antithetic variates:
distribution.sample(size=6, rule="halton", antithetic=True).round(4)

# ## Moments: [Descriptive Statistics — ChaosPy documentation](https://chaospy.readthedocs.io/en/master/descriptives.html#descriptives)

# the variance is defined as follows:
distribution.mom(2) - distribution.mom(1)**2

# or:
cp.Var(distribution)

# ## Seeding

np.random.seed(1234)
distribution.sample(5).round(4)

distribution.sample(5).round(4)

# ## [Copulas — ChaosPy documentation](https://chaospy.readthedocs.io/en/master/distributions/copulas.html)

np.random.seed(1234)
dist = cp.Iid(cp.Uniform(), 2)
copula = cp.Gumbel(dist, theta=1.5)

copula

np.random.seed(1234)
sample = copula.sample(10000)

plt.clf()
sns.jointplot(x=sample[0], y=sample[1], kind="hex")

# ok, what now…


