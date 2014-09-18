#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name = 'WarpedLMM',
      version='0.1',
      author='Nicolo Fusi',
      author_email="fusi@microsoft.com",
      description=("Warped linear mixed model"),
      license="Apache 2.0",
      keywords="genetics GWAS",
      packages = ["warpedlmm", 'warpedlmm.testing', 'warpedlmm.util'],
      install_requires=['scipy>=0.13', 'numpy>=1.8', 'matplotlib>=1.2', 'nose', 'fastlmm'],
      classifiers=[
      "License :: OSI Approved :: BSD License"],
      )
