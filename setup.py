# switch to  setuptools?
# https://github.com/storborg/python-packaging

from setuptools import setup

setup(
    name='tgap_ng',
    version='0.0.0',
    packages = ['tgap_ng',],
    scripts=['bin/tgap-ng.py'],
    url='https://github.com/bmmeijers/tgap-ng',
    author='Martijn Meijers',
    author_email='b.m.meijers@tudelft.nl',

)
