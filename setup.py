# -*- coding: utf-8 -*-

"""
Setup file for the *CMasher* package.
"""


# %% IMPORTS
# Built-in imports
from codecs import open
import re

# Package imports
from setuptools import find_packages, setup


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


# Setup function declaration
setup(name="prfpytools",
      version='0.0.dev0',
      author="Marco Aqil",
      author_email='marco.aqil@gmail.com',
      description=("prfpytools: a toolbox and helper package for prfpy"),
      long_description=long_description,
      url="https://github.com/VU-Cog-Sci/prfpytools",
      project_urls={
          'Documentation': "https://github.com/VU-Cog-Sci/prfpytools",
          'Source Code': "https://github.com/VU-Cog-Sci/prfpytools",
          },
      license='BSD-3',
      platforms=['Windows', 'Mac OS-X', 'Linux', 'Unix'],
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: BSD License',
          'Natural Language :: English',
          'Operating System :: MacOS',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: Unix',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Utilities',
          'Framework :: Matplotlib',
          ],
      python_requires='>=3.6, <4',
      packages=find_packages(),
      install_requires=requirements,
      zip_safe=False,
      )