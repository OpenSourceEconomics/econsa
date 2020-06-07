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

# +
import numpy as np
import pandas as pd
import scipy.stats as stats
import chaospy as cp
import matplotlib.pyplot as plt
import seaborn as sns

# import numba as nb

# from pathlib import Path
# from multiprocessing import Pool
# -

from temfpy.uncertainty_quantification import eoq_model

# # Harris Model from [Sensitivity analysis: A review of recent advances](https://www.sciencedirect.com/science/article/abs/pii/S0377221715005469)

# +
# function for constructing fig. 4:


def eoq_model_partial(x, r=0.1, fix_num=0):
    """
    Calculate the value of eoq_harris,
    fixing one x.
    
    Args: 
        params (np.array): 1d numpy array,
                           cuurrently only need the first param,
                           which is interest & depreciation rate, r=10.
        x (np.array or list): 2d numpy array with the independent variables,
                              currently only need the first 3 columns.
        fix_num (int): take value of 0~n-1.
    Output:
        y (np.array): 2d numpy array with the dependent variables,
                      keeping the fix_num-th x fixed.
    """

    x_np = np.array(x)

    y = np.zeros(shape=(x_np.T.shape[0], x_np.T.shape[0]))

    if fix_num == 0:
        for i, x_i in enumerate(x_np[fix_num]):
            y[i] = np.sqrt((24 * r * x_i * x_np[2]) / x_np[1])
    elif fix_num == 1:
        for i, x_i in enumerate(x_np[fix_num]):
            y[i] = np.sqrt((24 * r * x_np[0] * x_np[2]) / x_i)
    elif fix_num == 2:
        for i, x_i in enumerate(x_np[fix_num]):
            y[i] = np.sqrt((24 * r * x_np[0] * x_i) / x_np[1])
    return y


# -

# ## Data Generation

# +
# Set flags

seed = 1234
n = 10000

x_min_multiplier = 0.9
x_max_multiplier = 1.1

m_0 = 1230
c_0 = 0.0135
s_0 = 2.15
# -

params = np.zeros(shape=(1, 1))
params[0, 0] = 10

x_min_multiplier * m_0, x_max_multiplier * m_0

# ### No Monte Carlo

# +
np.random.seed(seed)

m = np.random.uniform(low=x_min_multiplier * m_0, high=x_max_multiplier * m_0, size=n)
c = np.random.uniform(low=x_min_multiplier * c_0, high=x_max_multiplier * c_0, size=n)
s = np.random.uniform(low=x_min_multiplier * s_0, high=x_max_multiplier * s_0, size=n)

y = eoq_model([m, c, s])

plt.clf()
sns.distplot(m)
# -

# ### Monte Carlo with `rvs`

# +
np.random.seed(seed)

m = stats.uniform(x_min_multiplier * m_0, x_max_multiplier * m_0).rvs(10000)
c = stats.uniform(x_min_multiplier * c_0, x_max_multiplier * c_0).rvs(10000)
s = stats.uniform(x_min_multiplier * s_0, x_max_multiplier * s_0).rvs(10000)

y = eoq_model([m, c, s])
# -

plt.clf()
sns.distplot(m)

plt.clf()
sns.distplot(y, hist_kws=dict(cumulative=True))

plt.clf()
sns.distplot(y)

# ### Monte Carlo with Chaospy (Closer to Borgonovoa & Plischkeb (2016))

sample_rule = "random"

# +
np.random.seed(seed)

m = cp.Uniform(x_min_multiplier * m_0, x_max_multiplier * m_0).sample(
    n, rule=sample_rule
)
c = cp.Uniform(x_min_multiplier * c_0, x_max_multiplier * c_0).sample(
    n, rule=sample_rule
)
s = cp.Uniform(x_min_multiplier * s_0, x_max_multiplier * s_0).sample(
    n, rule=sample_rule
)

y = eoq_model([m, c, s])
# -

df_monte_carlo = pd.DataFrame(data=[y, m, c, s])

# ## Graphs

# +
df_monte_carlo = pd.DataFrame(data=[y, m, c, s])

plt.clf()
fig, ax = plt.subplots(figsize=(6, 5))
ax = sns.heatmap(
    df_monte_carlo.T.corr(),
    vmin=-1,
    vmax=1,
    center=0,
    cmap="RdBu_r",
    annot=True,
    fmt=".2f",
)
ax.set_xticklabels(["y", "m", "c", "s"])
ax.set_yticklabels(["y", "m", "c", "s"])
plt.show()
# -

# ### Fig. 4

y_fix_m = eoq_model_partial([m, c, s], fix_num=0)

y_fix_m.shape

# +
# don't try at home:
# -

plt.clf()
# sns.set_palette(cubehelix)
for item in y_fix_m:
    sns.kdeplot(item)

# # Harris: Correlated Sampling: Copula

# ## Generation

# - `.Nataf(dist, R, ordering=None)`
# - `.TCopula(dist, df, R)`
# - `.AliMikhailHaq(dist, theta=0.5, eps=1e-06)`

# +
m_uniform = cp.Uniform(x_min_multiplier * m_0, x_max_multiplier * m_0)
c_uniform = cp.Uniform(x_min_multiplier * c_0, x_max_multiplier * c_0)
s_uniform = cp.Uniform(x_min_multiplier * s_0, x_max_multiplier * s_0)

x_uniform = cp.J(m_uniform, c_uniform, s_uniform)
# -

R = [[1, 0.5, 0.4], [0.5, 1, 0.7], [0.4, 0.7, 1]]

x_copula = cp.Nataf(x_uniform, R)

np.random.seed(seed)
x_copula_sample = x_copula.sample(n)

y_copula = eoq_model(x_copula_sample)

# ## Graphs

df_chaospy_copula = pd.DataFrame(
    data=[y_copula, x_copula_sample[0], x_copula_sample[1], x_copula_sample[2]]
)

plt.clf()
fig, ax = plt.subplots(figsize=(6, 5))
ax = sns.heatmap(
    df_chaospy_copula.T.corr(),
    vmin=-1,
    vmax=1,
    center=0,
    cmap="RdBu_r",
    annot=True,
    fmt=".2f",
)
ax.set_xticklabels(["y", "m", "c", "s"])
ax.set_yticklabels(["y", "m", "c", "s"])
plt.show()

# ## X & y

plt.clf()
sns.jointplot(x=x_copula_sample[0], y=y_copula, kind="hex")

plt.clf()
sns.jointplot(x=x_copula_sample[1], y=y_copula, kind="hex")

plt.clf()
sns.jointplot(x=x_copula_sample[2], y=y_copula, kind="hex")

# # Harris: Correlated Sampling: Rosenblatt: stats

# According to [Introducing Copula in Monte Carlo Simulation - Towards Data Science](https://towardsdatascience.com/introducing-copula-in-monte-carlo-simulation-9ed1fe9f905).

# +
x_normal = stats.multivariate_normal(mean=[m_0, s_0, c_0], cov=R)

np.random.seed(seed)
x_normal_sample = x_normal.rvs(n)
# -

x_uniform = stats.norm(loc=[m_0, s_0, c_0])
x_uniform_sample = x_uniform.cdf(x_normal_sample)

m_uniform = stats.uniform.ppf(
    x_uniform_sample[:, 0],
    loc=x_min_multiplier * m_0,
    scale=(x_max_multiplier - x_min_multiplier) * m_0,
)

s_uniform = stats.uniform.ppf(
    x_uniform_sample[:, 1],
    loc=x_min_multiplier * s_0,
    scale=(x_max_multiplier - x_min_multiplier) * s_0,
)

c_uniform = stats.uniform.ppf(
    x_uniform_sample[:, 2],
    loc=x_min_multiplier * c_0,
    scale=(x_max_multiplier - x_min_multiplier) * c_0,
)

y_rosenblatt = eoq_model([m_uniform, s_uniform, c_uniform])

# ## Graphs

# +
df_stats_rosenblatt = pd.DataFrame(data=[y_rosenblatt, m_uniform, s_uniform, c_uniform])

plt.clf()
fig, ax = plt.subplots(figsize=(6, 5))
ax = sns.heatmap(
    df_stats_rosenblatt.T.corr(),
    vmin=-1,
    vmax=1,
    center=0,
    cmap="RdBu_r",
    annot=True,
    fmt=".2f",
)
ax.set_xticklabels(["y", "m", "c", "s"])
ax.set_yticklabels(["y", "m", "c", "s"])
plt.show()
# -

plt.clf()
sns.jointplot(x=m_uniform, y=y_rosenblatt, kind="hex")

plt.clf()
sns.jointplot(x=s_uniform, y=y_rosenblatt, kind="hex")

plt.clf()
sns.jointplot(x=c_uniform, y=y_rosenblatt, kind="hex")

# # Harris: Correlated Sampling: Rosenblatt: chaospy

x_normal = cp.MvNormal(loc=[m_0, s_0, c_0], scale=R)

cp.Cov(x_normal)

# +
m_uniform = cp.Uniform(x_min_multiplier * m_0, x_max_multiplier * m_0)
c_uniform = cp.Uniform(x_min_multiplier * c_0, x_max_multiplier * c_0)
s_uniform = cp.Uniform(x_min_multiplier * s_0, x_max_multiplier * s_0)

x_uniform = cp.J(m_uniform, c_uniform, s_uniform)
# -

cp.Cov(x_uniform)

np.random.seed(seed)
x_normal_sample = x_normal.sample(n, rule="hammersley")

x_rosenblatt = x_uniform.inv(x_normal.fwd(x_normal_sample))
x_rosenblatt

y_rosenblatt = eoq_model(x_rosenblatt)

plt.clf()
sns.jointplot(x=x_normal_sample[0], y=x_normal_sample[1], kind="hex")

# ## Graphs

# What's wrong with this????

# +
df_chaospy_rosenblatt = pd.DataFrame(
    data=[y_rosenblatt, x_rosenblatt[0], x_rosenblatt[1], x_rosenblatt[2]]
)

plt.clf()
fig, ax = plt.subplots(figsize=(6, 5))
ax = sns.heatmap(
    df_chaospy_rosenblatt.T.corr(),
    vmin=-1,
    vmax=1,
    center=0,
    cmap="RdBu_r",
    annot=True,
    fmt=".2f",
)
ax.set_xticklabels(["y", "m", "c", "s"])
ax.set_yticklabels(["y", "m", "c", "s"])

plt.show()
# -

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
distribution.mom(2) - distribution.mom(1) ** 2

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
