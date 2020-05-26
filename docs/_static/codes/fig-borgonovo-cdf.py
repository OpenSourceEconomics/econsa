"""Plot uncertainty propagation.

We replicate and expand the figure 2 in [BP2016]_ that shows the uncertainty propagation in
Harris EOQ model. By propagating uncertainty in input variables, the output variable y becomes a
random variable.

.. [BP2016] Borgonovo, E., & Plischke, E. (2016). Sensitivity analysis: A review of recent
advances. European Journal of Operational Research, 248(3), 869–887.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import chaospy as cp
import numpy as np

from temfpy.uncertainty_quantification import eoq_harris


seed = 123
n = 10000

x_min_multiplier = 0.9
x_max_multiplier = 1.1
m_0 = 1230
s_0 = 0.0135
c_0 = 2.15

r = 0.1

np.random.seed(seed)
m = cp.Uniform(x_min_multiplier * m_0, x_max_multiplier * m_0).sample(n, rule="random")
s = cp.Uniform(x_min_multiplier * s_0, x_max_multiplier * s_0).sample(n, rule="random")
c = cp.Uniform(x_min_multiplier * c_0, x_max_multiplier * c_0).sample(n, rule="random")
x = np.array([m, s, c])

y = eoq_harris(x, r)


# Theoretical density for x
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

h_m = 1 / (x_max_multiplier * m_0 - x_min_multiplier * m_0)
ax[0].plot([x_min_multiplier * m_0, x_max_multiplier * m_0], [h_m, h_m], linewidth=2)
ax[0].fill(
    [
        x_min_multiplier * m_0,
        x_min_multiplier * m_0,
        x_max_multiplier * m_0,
        x_max_multiplier * m_0,
    ],
    [0, h_m, h_m, 0],
    alpha=0.5,
)
ax[0].set_ylim(bottom=0)
ax[0].set_xlabel(r"$X_1$")

h_s = 1 / (x_max_multiplier * s_0 - x_min_multiplier * s_0)
ax[1].plot([x_min_multiplier * s_0, x_max_multiplier * s_0], [h_s, h_s], linewidth=2)
ax[1].fill(
    [
        x_min_multiplier * s_0,
        x_min_multiplier * s_0,
        x_max_multiplier * s_0,
        x_max_multiplier * s_0,
    ],
    [0, h_s, h_s, 0],
    alpha=0.5,
)
ax[1].set_ylim(bottom=0)
ax[1].set_xlabel(r"$X_2$")

h_c = 1 / (x_max_multiplier * c_0 - x_min_multiplier * c_0)
ax[2].plot([x_min_multiplier * c_0, x_max_multiplier * c_0], [h_c, h_c], linewidth=2)
ax[2].fill(
    [
        x_min_multiplier * c_0,
        x_min_multiplier * c_0,
        x_max_multiplier * c_0,
        x_max_multiplier * c_0,
    ],
    [0, h_c, h_c, 0],
    alpha=0.5,
)
ax[2].set_ylim(bottom=0)
ax[2].set_xlabel(r"$X_3$")

fig.tight_layout()
fig.savefig("fig-borgonovo-cdf-x")


# Empirical density for y
fig, ax = plt.subplots()
sns.distplot(y)
ax.set_xlabel(r"$y$")

fig.tight_layout()
fig.savefig("fig-borgonovo-cdf-y")
