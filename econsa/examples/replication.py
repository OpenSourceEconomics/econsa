from morris import elementary_effects
import numpy as np
import pandas as pd


# inputs for example 1
def model_func(params):
    return params["value"].sum()

names = ["x1", "x2", "x3"]
params = pd.DataFrame(
    columns=["value"], data=[0, 0., 0], index=names)

cov = pd.DataFrame(
    data=[[1, 0.9, 0.4], [0.9, 1, 0.01], [0.4, 0.01, 1]],
    columns=names,
    index=names,
)
n_draws = 10000


if __name__ == "__main__":
    np.random.seed(12345)
    res = elementary_effects(
        model_func, params, cov, n_draws, sampling_scheme="sobol",
    )
    res = pd.DataFrame(res).round(2).rename(columns={"mu_ind": "abs_ee_ind", "sigma_ind": "sd_ee_ind"})
    res.to_pickle("our_example_1_radial.pickle")
