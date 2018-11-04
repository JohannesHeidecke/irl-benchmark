from setuptools import setup

setup(
    name='irl-benchmark',
    version='0.1.0',
    packages=['irl_benchmark', 'irl_benchmark.rl', 'irl_benchmark.rl.algorithms', 'irl_benchmark.irl',
              'irl_benchmark.irl.reward', 'irl_benchmark.irl.collect', 'irl_benchmark.irl.feature',
              'irl_benchmark.irl.algorithms', 'irl_benchmark.utils', 'irl_benchmark.config', 'irl_benchmark.metrics',
              'irl_benchmark.experiment'],
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
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    include_package_data=True
)
