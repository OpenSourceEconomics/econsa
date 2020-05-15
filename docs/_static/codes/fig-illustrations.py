from functools import partial

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


np.random.seed(123)


"""We create an illustration that shows the properties of the distribution we are interested in.
We choose a skewed normal distribution to already point to the fact that the median is not equal
to the mean.
"""
rv = stats.skewnorm(a=6)

confi_lower, median, confi_upper = rv.ppf([0.025, 0.5, 0.975])
plot_lower, plot_upper = rv.ppf([0.0001, 0.999])
mean, std = rv.mean(), rv.std()

x = np.linspace(plot_lower, plot_upper, 100)

fig, ax = plt.subplots()

ax.plot(x, rv.pdf(x))

x = np.linspace(confi_lower, confi_upper, 100)
y = rv.pdf(x)

ax.fill_between(x, y, alpha=0.2, label="95% CI")

p_axvline = partial(ax.axvline, linestyle="--", color="lightgrey")
[p_axvline(arg) for arg in [mean, median, mean - std, mean + std]]

pos = (mean - std, median, mean, mean + std)
lab = (r"$-\sigma$", r"$\eta$", r"$\mu$", r"$+\sigma$")
ax.set_xticks(pos), ax.set_xticklabels(lab)
ax.set_xlabel(r"$\Delta$ Schooling")

ax.set_yticklabels([]), ax.set_ylabel(r"Density")
ax.set_ylim([0, None])

ax.legend()

fig.savefig("fig-illustration-density")

""" We create a simple illustration that shows the basic idea that we choose a particular policy
that allows to attain a level of schooling with a given degree of uncertainty.
"""
fig, ax = plt.subplots()


rv = stats.skewnorm(a=6)

confi_lower, median, confi_upper = rv.ppf([0.025, 0.5, 0.975])
std = rv.std()
mean = rv.mean()

plot_lower, plot_upper = rv.ppf([0.0001, 0.999])

x = np.linspace(-1, 1, 1000)
y = rv.pdf(x)

ax.plot(x, y, label="Subsidy A")
ax.plot(x + 0.1, y, label="Subsidy B")

ax.set_ylim([0, 0.75])
ax.set_xlim([-0.4, 0.2])

ax.axvline(0.0, linestyle="--", color="lightgrey")
ax.axes.get_xaxis().set_ticklabels([])

labels = ax.get_xticklabels()
labels[4] = r"$\xi$"
ax.axes.get_xaxis().set_ticklabels(labels)

ax.axes.get_yaxis().set_ticklabels([])
ax.set_ylabel(r"Density")

ax.set_xlabel(r"$\Delta$ Schooling")

ax.legend()

fig.savefig("fig-illustration-reliability")
