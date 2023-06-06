"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

from setuptools import setup, find_packages, Extension
import sys

if sys.version_info.major != 3:
    raise RuntimeError('RP-KMeans requires Python 3')


setup(
    name='aide',
    description='Autoencoder-imputed distance-preserved embedding (AIDE), a dimension reduction algorithms that combines both Multidimensional Scaling (MDS) and AutoEncoder (AE).',
    version='1.0.0',
    author='Yu Huang',
    author_email='yuhuang-cst@foxmail.com',
    packages=['aide'],
    zip_safe=False,
    url='https://github.com/yuhuang-cst/aide',
    license='LICENSE',
    long_description=open('README.md').read(),
    install_requires=open('requirements.txt').read(),
)


