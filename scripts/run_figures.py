#!/usr/bin/env python
"""Create figures.

This script supports our CI setup as it simply runs all files that create figures for our
documentation.

"""
import glob
import os
import subprocess as sp

if __name__ == "__main__":

    os.chdir(os.environ["PROJECT_ROOT"] + "/docs/_static/codes")

    [sp.check_call(f"python {fname}", shell=True) for fname in glob.glob("fig-*.py")]
