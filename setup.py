#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os
from io import open

install_requires = [
    "biopython>=1.78",
    "scikit-learn==0.21.3",
    "pandas==0.24.1",
    "numpy==1.16.1",
    "keras==2.2.4",
    "tensorflow==1.15.4",
    "matplotlib==2.2.3",
    "appdirs>=1.4.3",
    "scipy==1.2.0",
    "protobuf<4",
]

about = {}
here = os.path.abspath(os.path.dirname(__file__))

# Read version number from plantbgc/__version__.py
with open(os.path.join(here, "plantbgc", "__version__.py"), encoding="utf-8") as f:
    exec(f.read(), about)

# Read README
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="plantbgc",
    version=about["__version__"],
    description="PlantBGC: Transformer-based biosynthetic gene cluster candidate detection for plant genomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("test", "test.*")),
    author="PlantBGC Authors",
    author_email="yzhao66@ncsu.edu",
    license="MIT",
    python_requires=">=3.6,<3.8",
    install_requires=install_requires,
    extras_require={
        "hmm": ["hmmlearn>=0.2.1,<0.2.7"],
    },
    keywords=["bioinformatics", "BGC", "transformer", "plants"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
    ],
    url="https://github.com/Yuhanzhao-233/PlantBGC",
    entry_points={
        "console_scripts": [
            "plantbgc=plantbgc.main:main",
        ]
    },
)
