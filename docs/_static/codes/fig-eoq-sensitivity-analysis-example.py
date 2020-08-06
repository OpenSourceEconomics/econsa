import matplotlib.pyplot as plt
import numpy as np
from temfpy.uncertainty_quantification import eoq_model

# TODO: This code will be reactivated once we have Tim's PR merged.
# from econsa.kucherenko import kucherenko_indices
#
# def eoq_model_transposed(x):
#     """EOQ Model but with variables stored in columns."""
#     return eoq_model(x.T)
#
# mean = np.array([1230, 0.0135, 2.15])
# cov = np.diag([1, 0.000001, 0.01])
#
# indices = kucherenko_indices(
#     func=eoq_model_transposed,
#     sampling_mean=mean,
#     sampling_cov=cov,
#     n_draws=1_000_000,
#     sampling_scheme="sobol")
#
# sobol_first = indices.loc[(slice(None), "first_order"), "value"].values
# sobol_total = indices.loc[(slice(None), "total"), "value"].values
#
# x = np.arange(3)  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, sobol_first, width, label='First-order')
# rects2 = ax.bar(x + width/2, sobol_total, width, label='Total')
#
# ax.set_ylim([0, 1])
# ax.legend()
#
# ax.set_xticks(x)
# ax.set_xticklabels(["$x_0$", "$x_1$", "$x_2$"])
# ax.legend()
#
# fig.savefig("fig-eoq-sensitivity-analysis-sobol")
