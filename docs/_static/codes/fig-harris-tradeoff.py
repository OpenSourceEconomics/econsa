"""We create replicate the figure 1 in [H1990]_ that shows the trade-off in Harris EOQ model,
where an increase in the size of order results in a decrease in set-up costs, but an increase in
interest & depreciation cost.

.. [H1990] Harris, F. W. (1990). How Many Parts to Make at Once. Operations Research, 38(6),
947–950.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def eoq_harris_total_cost(x, y, r=10):
    r"""Economic order quantity model.

    This function computes the total costs of economic order quantity model, based on the model
    presented in [H1990]_

    For plotting convenience, the total cost here excludes the last :math:`c`, since it is
    assumed to be constant, as in Harris (1990).

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
    .. [H1990] Harris, F. W. (1990). How Many Parts to Make at Once. Operations Research, 38(6),
    947–950.
    """

    m, s, c = x

    y_np = np.array(y)

    t_setup = (1 / y_np) * s

    t_interest = 1 / (24 * r * m) * (y_np * c + s)

    t = t_setup + t_interest

    return t_setup, t_interest, t


m, c, s = 1000, 0.1, 2

y = np.arange(300, 5200, 1)
x = [m, s, c]

t_setup, t_interest, t = eoq_harris_total_cost(x, y)


# Start plotting
fig, ax = plt.subplots()

ax.plot(y, t_setup, label="Setup")
ax.plot(y, t_interest, label="Capital")
ax.plot(y, t, label="Total")
ax.axvline(2190, linestyle="--", color="lightgrey")

ax.axes.get_yaxis().set_ticklabels([])
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))

# TODO: Let's add X^* to the x-axis., then update the figure for the docs.

ax.legend()

ax.set_xlabel("Size of order")
ax.set_ylabel("Cost")

fig.savefig("fig-harris-tradeoff")
