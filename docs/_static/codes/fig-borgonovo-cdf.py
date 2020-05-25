"""We replicate and expand the figure 2 in [BP2016]_ that shows the uncertainty propagation in
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

# Start plotting

# TODO: Let's just use the theoretical distribution for the input parameters and have a separate
#  figure for the y.
plt.clf()
fig, ax = plt.subplots(2, 2)

sns.distplot(m, ax=ax[0, 0])
ax[0, 0].set_xlabel(r"$x_1$")

sns.distplot(s, ax=ax[0, 1])
ax[0, 1].set_xlabel(r"$x_2$")

sns.distplot(c, ax=ax[1, 0])
ax[1, 0].set_xlabel(r"$x_3$")

sns.distplot(y, ax=ax[1, 1])
ax[1, 1].set_xlabel(r"$y$")

fig.tight_layout()
fig.savefig("fig-borgonovo-cdf")
