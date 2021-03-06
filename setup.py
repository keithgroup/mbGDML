#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'natsort', 'cclib>=1.7', 'numpy', 'dscribe', 'ase', 'sgdml', 'mako',
    'matplotlib'
]

setup_requirements = [ ]

test_requirements = requirements.append(['pytest'])

setup(
    author="Alex M. Maldonado",
    author_email='aalexmmaldonado@gmail.com',
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7'
    ],
    description="Many-body implementation of symmetric  domain machine learning force fields",
    install_requires=requirements,
    extras_require={},
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='mbgdml',
    name='mbgdml',
    packages=find_packages(include=['mbgdml', 'mbgdml.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/aalexmmaldonado/mbGDML',
    version='0.0.1',
    zip_safe=False,
)
