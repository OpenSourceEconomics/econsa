import os

n_threads = 1
os.environ["NUMBA_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}"

import respy as rp
from estimagic.differentiation.differentiation import jacobian
from estimagic.inference.likelihood_covs import cov_jacobian
from pathlib import Path
import pandas as pd
import numpy as np
from morris import elementary_effects
from time import time
from joblib import wrap_non_picklable_objects
from joblib.externals.loky import set_loky_pickler


start_params, options, data = rp.get_example_model("kw_94_one", with_data=True)
start_params = pd.read_csv("params.csv").set_index(["category", "name"])

options["simulation_agents"] = 4000

to_drop = [
    ('lagged_choice_1_edu', 'probability'),
    ('initial_exp_edu_10', 'probability'),
    ('maximum_exp', 'edu')
]
cov_path = Path("bld/cov.pickle")
if cov_path.exists():
    cov = pd.read_pickle(cov_path)
else:
    loglikeobs = rp.get_crit_func(
        start_params, options, data, return_scalar=False)
    jac = jacobian(loglikeobs, start_params, extrapolation=False)
    reduced_jac = jac.drop(columns=to_drop)
    cov = cov_jacobian(reduced_jac)
    pd.to_pickle(cov, cov_path)
se = np.sqrt(np.diagonal(cov))
start_params["se"] = se.tolist() + [np.nan] * 3

cov_df = pd.DataFrame(cov, columns=start_params.index[:-3], index=start_params.index[:-3])

print("Jacobian done")

simfunc = rp.get_simulate_func(start_params, options)

to_append = start_params.loc[to_drop]

def qoi(params):
    p1 = pd.concat([params, to_append])
    p2 = p1.copy(deep=True)
    p2.loc[("nonpec_edu", "constant"), "value"] += 500
    df1 = simfunc(p1)
    df2 = simfunc(p2)
    return df2["Experience_Edu"].mean() - df1["Experience_Edu"].mean()

np.random.seed(5471)

start_params_short = start_params.drop(to_drop)

res = elementary_effects(qoi, start_params_short, cov_df, n_draws=10000, sampling_scheme="random", n_cores=30)

# res = pd.DataFrame(res)
pd.to_pickle(res, "bld/indices/kw_indices.pickle")



