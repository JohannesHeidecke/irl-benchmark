from setuptools import setup
from setuptools import find_packages

setup(name='irl-benchmark',
      version='0.0.1',
      description='IRL Benchmarking',
      author='Johannes Heidecke',
      install_requires=['gym', 'numpy', 'cvxpy'],
      packages=find_packages())
