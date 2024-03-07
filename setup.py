"""Setup file for pip package"""

from setuptools import find_packages, setup

setup(
    name="xrdanalysis",
    version="0.1.0",
    description="Repository for collaboration of EOSDX team",
    packages=find_packages("src"),
    package_dir={"": "src"},
    author="EOSDX",
    license="",
)
