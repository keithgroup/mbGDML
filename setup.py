#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'natsort>=7.0.1', 'cclib>=1.6.2',
                'periodictable>=1.5.2', 'mako>=1.1.2']

setup_requirements = [ ]

test_requirements = ['pytest>=5.4.1']

setup(
    author="Alex M. Maldonado",
    author_email='alex.maldonado113@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Many-body implementation of symmetric gradient domain machine learning",
    entry_points={
        'console_scripts': [
            'mbgdml=mbgdml.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
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
