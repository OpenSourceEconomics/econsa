import matplotlib.pyplot as plt
import numpy as np


"""We create replicate the figure 1 in [H1990]_
that shows the trade-off in Harris EOQ model,
where an increase in the size of order results in a decrease in set-up costs,
but an increase in interest & depreciation cost.

.. [H1990] Harris, F. W. (1990).
        How Many Parts to Make at Once.
        Operations Research, 38(6), 947–950.
"""

# Helper function


def eoq_harris_total_cost(x, y, r=10):
    r"""Economic order quantity model.
    This function computes the total costs of economic order quantity model,
    based on the model presented in [H1990]_

    For plotting convenience, the total cost here excludes the last :math:`c`,
    since it is assumed to be constant, as in Harris (1990).

    Parameters
    ----------
    x : array_like
        Core parameters of the model
    y : integer
        Order size
    r : float, optional
        Annual interest rate

    Returns
    -------
    t_setup : float
              Set-up cost
    t_interest : float
                 Interest and depreciation cost
    t : float
        Sum of `t_setup` and `t_interest`

    References
    ----------
    .. [H1990] Harris, F. W. (1990).
        How Many Parts to Make at Once.
        Operations Research, 38(6), 947–950.
    """

    m, s, c = x

    y_np = np.array(y)

    t = np.zeros(y.shape)
    t_setup = np.zeros(y.shape)
    t_setup = (1 / y_np) * s

    t_interest = np.zeros(y.shape)
    t_interest = 1 / (24 * r * m) * (y_np * c + s)

    t = t_setup + t_interest

    return (t_setup, t_interest, t)


# Data generation

y = np.arange(300, 5200, 1)

m = 1000
c = 0.1
s = 2

x = np.array([m, s, c])

t_setup, t_interest, t = eoq_harris_total_cost(x, y)

# Start plotting

plt.clf()
fig, ax = plt.subplots()

ax.plot(y, t_setup, label="Set-up Cost")
ax.plot(y, t_interest, label="Interest and Depreciation Cost")
ax.plot(y, t, label="Total Cost")
ax.axvline(2190, linestyle="--", color="lightgrey")

ax.axes.get_yaxis().set_ticklabels([])

ax.legend()

ax.set_xlabel("Size of Order")
ax.set_ylabel("Costs")

fig.savefig("fig-harris-tradeoff")
