import os
import sys

from setuptools import setup
from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', 'Arguments to pass to py.test')]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ['--cov=irl_benchmark', '--cov-report=html', '--cov-report=term', '--cov-report=xml']

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)

setup(
    name='irl-benchmark',
    version='0.1.1dev',
    packages=['irl_benchmark', 'irl_benchmark.rl', 'irl_benchmark.rl.algorithms', 'irl_benchmark.irl',
              'irl_benchmark.irl.reward', 'irl_benchmark.irl.collect', 'irl_benchmark.irl.feature',
              'irl_benchmark.irl.algorithms', 'irl_benchmark.utils', 'irl_benchmark.config', 'irl_benchmark.metrics',
              'irl_benchmark.experiment', 'irl_benchmark.envs'],
    url='https://github.com/JohannesHeidecke/irl-benchmark',
    license='LICENSE',
    author='Anton Osika, Adria Garriga-Alonso, Johannes Heidecke, Max Daniel, Sayan Sarkar',
    author_email='jheidecke@gmail.com',
    description='Benchmark framework for inverse reinforcement learning algorithms.',
    install_requires=[
        'gym>=0.10.5',
        'cvxpy>=1.0.9',
        'numpy>=1.15.0',
        'typing>=3.6.4',
        'setuptools>=40.0.0',
        'comet_ml>=1.0.31',
        'msgpack_python>=0.5.6',
        'msgpack_numpy>=0.4.4.1',
        'torch>=0.4.0',
        'tqdm>=4.28.1',
        'sparse>=0.5.0',
        'pathfinding>=0.0.2',
    ],
    test_suite='test',
    cmdclass={'test': PyTest},
    include_package_data=True
)
