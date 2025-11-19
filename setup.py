"""
Setup script for Optimization and Probabilistic Modeling package.
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='optimization-probabilistic-modeling',
    version='0.1.0',
    author='Mohammad Reza Askari',
    author_email='maskari@hawk.iit.edu',
    description='A comprehensive CPU/GPU accelerated Python package for probabilistic programming, Bayesian inference, and optimization',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rezaaskary/Optimization-and-probabilistic-modeling',
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.9',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=22.0',
            'flake8>=4.0',
            'mypy>=0.990',
        ],
        'docs': [
            'sphinx>=4.5',
            'sphinx-rtd-theme>=1.0',
            'numpydoc>=1.4',
        ],
        'viz': [
            'matplotlib>=3.5',
            'seaborn>=0.12',
            'plotly>=5.0',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            # Add command-line tools here if needed
        ],
    },
    keywords='optimization probabilistic-programming bayesian-inference mcmc jax',
    project_urls={
        'Bug Reports': 'https://github.com/rezaaskary/Optimization-and-probabilistic-modeling/issues',
        'Source': 'https://github.com/rezaaskary/Optimization-and-probabilistic-modeling',
        'Documentation': 'https://github.com/rezaaskary/Optimization-and-probabilistic-modeling#readme',
    },
)
