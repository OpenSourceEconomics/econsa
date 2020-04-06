from setuptools import find_packages
from setuptools import setup

p = find_packages()
print(p)

setup(
    name="econsa",
    version="0.1.5",
    description=(
        "Conda installable package comprising of a python toolbox for uncertainty quantification and sensitivity analysis tailored to economic models."
    ),
    license="MIT",
    url="https://github.com/OpenSourceEconomics/econsa",
    author="OpenSourceEconomics",
    author_email="linda.maokomatanda@gmail.com",
    packages=p,
    zip_safe=False,
    package_data={
        "utilities": [
            ]
    },
    include_package_data=True,
)
