#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

# TODO: Scipy must be less than 1.8 because of a bayesian-optimization bug.
# This is fixed in https://github.com/fmfn/BayesianOptimization/commit/35535c6312f365ead729de3d889d7b1fae1a8e0b
# but not released yet.
requirements = [
    'ase', 'bayesian-optimization', 'cclib>=1.7', 'matplotlib', 'natsort',
    'numpy', 'scipy<=1.7.3', 'umap-learn'
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
