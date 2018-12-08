[![CircleCI](https://circleci.com/gh/JohannesHeidecke/irl-benchmark.svg?style=svg)](https://circleci.com/gh/JohannesHeidecke/irl-benchmark) [![Maintainability](https://api.codeclimate.com/v1/badges/f929f0f865714080daf6/maintainability)](https://codeclimate.com/github/JohannesHeidecke/irl-benchmark/maintainability) [![Test Coverage](https://api.codeclimate.com/v1/badges/f929f0f865714080daf6/test_coverage)](https://codeclimate.com/github/JohannesHeidecke/irl-benchmark/test_coverage)

# Introduction

`irl-benchmark` is a modular library for evaluating various **Inverse Reinforcement Learning** algorithms. It provides an extensible platform for experimenting with different environments, algorithms and metrics. 

# Installation

`conda create --name irl-benchmark python=3.6`

`source activate irl-benchmark`

`pip install -r requirements.txt`

# Getting Started

Start by generating expert data by

`python generate_expert_data.py`

Then run 

`python main.py` 

to get an overview of how all the components of `irl-benchmark` work together.

# Documentation

Documentation is available _as work in progress_ at: [https://johannesheidecke.github.io/irl-benchmark](https://johannesheidecke.github.io/irl-benchmark).

You may find the [extending](https://johannesheidecke.github.io/irl-benchmark) part useful if you are planning to author new algorithms.

# Environemts

- [FrozenLake-v0](https://gym.openai.com/envs/FrozenLake-v0/)

- [FrozenLake8x8-v0](https://gym.openai.com/envs/FrozenLake8x8-v0/)

# Algorithms

- [Apprenticeship Learning (SVM Based)](http://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)
- Apprenticeship Learning (Projection Based)
- [Maximum Entropy IRL](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf) 
- [Maximum Causal Entropy IRL](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf)

# Metrics


Copyright: Adria Garriga-Alonso, Anton Osika, Johannes Heidecke, Max Daniel, and Sayan Sarkar.
