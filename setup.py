#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

setup(name = 'WarpedLMM',
      version='0.1',
      author='Nicolo Fusi',
      author_email="fusi@microsoft.com",
      description=("Warped linear mixed model"),
      license="BSD 3-clause",
      keywords="genetics GWAS",
      packages = ["warpedlmm.fastlmm", 'warpedlmm.testing', 'warpedlmm.util'],
      install_requires=[], # pandas, scipy, numpy
      classifiers=[
      "License :: OSI Approved :: BSD License"],
      )
