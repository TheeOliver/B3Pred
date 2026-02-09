"""
Setup file for B3Pred - Blood-Brain Barrier Permeability Prediction
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the long description from README if it exists
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')

setup(
    name="b3pred",
    version="1.0.0",
    author="Your Name",  # Update this
    author_email="your.email@example.com",  # Update this
    description="Graph Neural Network models for Blood-Brain Barrier permeability prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/B3Pred",  # Update this
    packages=find_packages(where='.', exclude=['tests', 'experiments', 'data']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torch-geometric>=2.0.0",
        "torch-scatter>=2.0.0",
        "torch-sparse>=0.6.0",
        "torch-cluster>=1.6.0",
        "torch-spline-conv>=1.2.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "rdkit>=2022.3.0",
        "optuna>=3.0.0",
        "cma>=3.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "logging": [
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        'console_scripts': [
            'b3pred-train=scripts.train:main',
            'b3pred-optimize=optimizations.bayesian_optimization:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.md', '*.txt'],
    },
    zip_safe=False,
)