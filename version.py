from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GPL-v3 License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "fitting normalization model using prfpy"
# Long description will go up on the pypi page
long_description = """

pRFpy normalization
========

"""

NAME = "prfpy-norm"
MAINTAINER = "Marco Aqil"
MAINTAINER_EMAIL = "m.aqil@spinozacentre.nl"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/VU-Cog-Sci/prfpy-norm"
DOWNLOAD_URL = ""
LICENSE = "GPL3"
AUTHOR = "Marco Aqil"
AUTHOR_EMAIL = "m.aqil@spinozacentre.nl"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'core': [pjoin('test', 'data', '*')]}
REQUIRES = ["prfpy"]
