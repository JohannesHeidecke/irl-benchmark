.. IRL Benchmark documentation master file, created by
   sphinx-quickstart on Mon Oct 22 01:53:59 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


:tocdepth: 4

Welcome to IRL Benchmark's documentation!
=========================================

Getting started:
----------------

You can find installation instructions :doc:`here<installation>`.

If you'd like to use this project for your own experiments or if you are interested in replicating some results, have a look at the :doc:`quickstart tutorial<quickstart>`.

See :doc:`extending the benchmark<extending>` if you want to make one of the following changes:

* adding a new environment
* adding a new IRL algorithm
* adding a new RL algorithm
* adding a new metric

If you want your changes to be available to the entire community, please fork the repository and make a pull request. General guidelines for this can be found in our :doc:`collaboration guide<collaboration>`.

About the project
-----------------

We want reusable resuts, reproducible results, robust results for IRL!

What should be **reusable**:

* environments used for testing
* potentially expert demonstrations (especially expensive if collected from human)
* IRL algorithm implementations
* RL algorithm implementations
* experiment code and metrics

This makes the experiments reproducible and the findings more robust

Reinforcement learning research suffers from a reproducibility crisis (highly recommended to watch `this ICLR talk by Joelle Pineau <https://youtu.be/Vh4H0gOwdIg?t=120>`_). This seems to be even worse in inverse reinforcement learning, where in addition we are often missing the used expert demonstrations and so far no one is really sure what the state of the art currently is.

This benchmark aims to contribute greatly towards solving this issue by creating a standard for reproducible IRL research.



Dive into the docs:
-------------------

Some important classes to know:

* :class:`BaseIRLAlgorithm<.irl.algorithms.base_algorithm.BaseIRLAlgorithm>`: all IRL algorithms extend this class and must overwrite its `train` method. It pre-processes any config when new algorithms are created to make sure values lie within allowed ranges and missing values are set to their default values. Additionally it provides some useful methods that are used by a variety of IRL algorithms, such as calculating empirical feature counts.
* TODO: add more

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   quickstart
   extending
   collaboration
   modules

Indices and Search:
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
