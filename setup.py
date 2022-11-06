#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

requirements = [
    'ase', 'cclib>=1.7', 'matplotlib', 'natsort', 'numpy', 'scipy', 'psutil',
    'bayesian-optimization>=1.3.0', 'pandas'
]
setup_requirements = [ ]
test_requirements = requirements.append(['pytest'])

setup(
    packages=find_packages(include=['mbgdml', 'mbgdml.*']),
    install_requires=requirements,
    include_package_data=True,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
