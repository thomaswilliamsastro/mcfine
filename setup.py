import os
from setuptools import setup

install_requires = [
    'astropy',
    'corner',
    'emcee',
    'lmfit',
    'matplotlib',
    'numdifftools',
    'numpy',
    'scipy',
    'threadpoolctl',
    'tomli',
    'tqdm',
]

# For readthedocs building, ignore ndradex since it doesn't compile:
if not os.getenv('READTHEDOCS'):
    install_requires.append('ndradexhyperfine')

if __name__ == "__main__":
    setup(install_requires=install_requires)
