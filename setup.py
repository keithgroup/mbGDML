#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import versioneer

requirements = [
    'ase', 'cclib>=1.7', 'numpy', 'scipy'
]

setup_requirements = [ ]

test_requirements = requirements.append(['pytest'])

setup(
    install_requires=requirements,
    extras_require={
        'all': [
            'natsort', 'mako', 'umap-learn', 'bayesian-optimization',
            'matplotlib'
        ]
    },
    include_package_data=True,
    packages=find_packages(include=['mbgdml', 'mbgdml.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
