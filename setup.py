#!/usr/bin/env python
"""
Setup script for Topological Photonic Crystal Optimizer

Install with: pip install -e .
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_file(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    req_file = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            return [line.strip() for line in f
                   if line.strip() and not line.startswith('#')]
    return []

setup(
    name="topological-photonic-optimizer",
    version="1.0.0",
    author="Sakeeb Rahman",
    author_email="rahman.sakeeb@gmail.com",
    description="Advanced ML framework for designing topological photonic crystal ring resonators",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/Sakeeb91/topological-photonic-crystal-optimizer",
    packages=find_packages(exclude=["tests", "tests.*", "results", "results.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ],
        'meep': [
            # MEEP is typically installed via conda
            # conda install -c conda-forge pymeep
        ],
    },
    entry_points={
        'console_scripts': [
            'topo-optimize=run_optimization:main',
            'topo-multi-obj=run_multi_objective_optimization:main',
            'topo-visualize=visualize_best_design:main',
        ],
    },
    include_package_data=True,
    package_data={
        'src': ['*.yaml'],
    },
    zip_safe=False,
    keywords=[
        'photonics',
        'topological physics',
        'machine learning',
        'bayesian optimization',
        'photonic crystals',
        'SSH model',
        'MEEP',
        'electromagnetic simulation',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/Sakeeb91/topological-photonic-crystal-optimizer/issues',
        'Source': 'https://github.com/Sakeeb91/topological-photonic-crystal-optimizer',
        'Documentation': 'https://github.com/Sakeeb91/topological-photonic-crystal-optimizer/blob/main/README.md',
    },
)
