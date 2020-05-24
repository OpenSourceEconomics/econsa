import numpy as np
import chaospy as cp
import matplotlib.pyplot as plt
import seaborn as sns

"""We replicate and expand the figure 2 in [BP2016]_
that shows the uncertainty propagation in Harris EOQ model.
By propagating uncertainty in input variables,
the output variable y becomes a random variable.

.. [BP2016] Borgonovo, E., & Plischke, E. (2016).
        Sensitivity analysis: A review of recent advances.
        European Journal of Operational Research, 248(3), 869–887.
"""

# Helper function


def eoq_harris(x, r=0.1):
    r"""Economic order quantity model.

    This function computes the optimal economic order quantity (EOQ) based on the model presented in
    [H1990]_. The EOQ minimizes the holding costs as well as ordering costs. The core parameters of
    the model are the units per months `x[0]`, the unit price of items in stock `x[1]`,
    and the setup costs of an order `x[2]`. The annual interest rate `r` is treated as an
    additional parameter.

    Parameters
    ----------
    x : array_like
        Core parameters of the model

    r : float, optional
        Annual interest rate

    Returns
    -------

    float
        Optimal order quantity

    Notes
    -----

    A historical perspective on the model is provided by [E1990]_. A brief description with the core
    equations is available in [W2020]_.

    References
    ----------

    .. [H1990] Harris, F. W. (1990).
        How Many Parts to Make at Once.
        Operations Research, 38(6), 947–950.

    .. [E1990] Erlenkotter, D. (1990).
        Ford Whitman Harris and the Economic Order Quantity Model.
        Operations Research, 38(6), 937–946.

    .. [W2020] Economic Order Quantity. (2020, April 3). In Wikipedia.
        Retrieved from
        `https://en.wikipedia.org/w/index.php\
        ?title=Economic_order_quantity&oldid=948881557 <https://en.wikipedia.org/w/index.php
        ?title=Economic_order_quantity&oldid=948881557>`_

    Examples
    --------

    >>> x = [1, 2, 3]
    >>> y = eoq_harris(x, r=0.1)
    >>> np.testing.assert_almost_equal(y, 12.649110640673518)
    """

    m, s, c = x
    y = np.sqrt((24 * m * s) / (r * c))

    return y


# Data generation

seed = 123
n = 10000

x_min_multiplier = 0.9
x_max_multiplier = 1.1
m_0 = 1230
s_0 = 0.0135
c_0 = 2.15

r = 0.1

np.random.seed(seed)
m = cp.Uniform(x_min_multiplier*m_0, x_max_multiplier*m_0).sample(n, rule="random")
s = cp.Uniform(x_min_multiplier*s_0, x_max_multiplier*s_0).sample(n, rule="random")
c = cp.Uniform(x_min_multiplier*c_0, x_max_multiplier*c_0).sample(n, rule="random")
x = np.array([m, s, c])

y = eoq_harris(x, r)

# Start plotting

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
