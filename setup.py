#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

# TODO: Bayesian optimization on PyPI is incompatible with scipy>1.8.x.
# We install bayesian-optimization from git until this is fixed.
requirements = [
    'ase', 'cclib>=1.7', 'matplotlib', 'natsort', 'numpy', 'scipy', 'psutil',
    'bayesian-optimization @ git+https://github.com/fmfn/BayesianOptimization'
]
setup_requirements = [ ]
test_requirements = requirements.append(['pytest'])

setup(
    install_requires=requirements,
    include_package_data=True,
    packages=find_packages(include=['mbgdml', 'mbgdml.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
