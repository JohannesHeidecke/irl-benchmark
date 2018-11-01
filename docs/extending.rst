Extending the benchmark
=======================

This guide provides information on how to extend the benchmark, e.g. to add a new IRL algorithm. We cover four possible extension: adding new environments, adding new IRL algorithms, adding new RL algorithms, and adding new metrics.

We are happy to add new extensions to the main benchmark. If you are interested in collaborating, please see our :doc:`collaboration guide<collaboration>`.

Environments
------------

``TODO``

``TODO``: also talk here about collecting expert trajectories. The expert trajectories need to contain features, potentially artificially wrapped added with FeatureWrapper?

IRL algorithms
--------------

All IRL algorithms have to extend the abstract base class :class:`BaseIRLAlgorithm<.irl.algorithms.base_algorithm.BaseIRLAlgorithm>`:

.. code-block:: python
    
    from irl_baselines.irl.algorithms.base_algorithm import BaseIRLAlgorithm

    class ExampleIRL(BaseIRLAlgorithm):
        """An example IRL algorithm"""

Initializing
^^^^^^^^^^^^

If your algorithm class implements it's own ``__init__`` method, make sure that the base class ``__init__`` is called as well. This is necessary since in this way the passed config is preprocessed correctly. Please use the same parameters for the new ``__init__`` and don't add additional ones. Any additional parameters required by your algorithm should go into the config dictionary.

 .. code-block:: python
    :emphasize-lines: 5
    
    def __init__(self, env: gym.Env, expert_trajs: List[Dict[str, list]],
                 rl_alg_factory: Callable[[gym.Env], BaseRLAlgorithm],
                 config: dict):
        """ docstring ... """
        super(ExampleIRL, self).__init__(env, expert_trajs, rl_alg_factory, config)


Let's go over the four parameters that are always passed to an IRL algorithm when it is created:

* ``env`` is an openAI gym environment, at least wrapped in a :class:`RewardWrapper<.irl_benchmark.irl.reward.reward_wrapper.RewardWrapper>`. The reward wrapper will make sure that the environment's true reward function is not accidentally leaked to the IRL algorithm. If required, the true reward can still be read from the info dictionary returned by the environments ``step`` function as follows:

 .. code-block:: python
    :emphasize-lines: 2
    
    state, reward, done, info = env.step(action)
    print(info['true_reward'])

* ``expert_trajs`` is a list of trajectories collected from the expert. Each trajectory is a dictionary with keys ``['states', 'actions', 'rewards', 'true_rewards', 'features']``. Each value in the dictionary is a list, containing e.g. all states ordered by time. The states list will have one more element than the others, since it contains both the initial and final state. In the case of expert trajectories, ``true_rewards`` will be an empty list. See :func:`collect_trajs<irl_benchmark.irl.collect.collect_trajs>` which defines how trajectories are generated.

* ``rl_alg_factory`` is a function which takes an environment and returns a newly initialized reinforcement learning algorithm. This is used to keep the IRL algorithms flexible about which concrete RL algorithm they can be used with. If your IRL algorithm requires a specific RL algorithm (such as in guided cost learning), simply overwrite ``self.rl_alg_factory`` in your ``__init__`` after calling the base class ``__init__``.

 .. code-block:: python
    :emphasize-lines: 7,8,9,10
    
    def __init__(self, env: gym.Env, expert_trajs: List[Dict[str, list]],
                 rl_alg_factory: Callable[[gym.Env], BaseRLAlgorithm],
                 config: dict):
        """ docstring ... """
        super(ExampleIRL, self).__init__(env, expert_trajs, rl_alg_factory, config)

        # enforce use of specific RL algorithm:
        def specific_rl_alg_factory(env: gym.Env):
            return SpecificRlAlg(env, {'hyperparam': 42})
        self.rl_alg_factory = specific_rl_alg_factory

* ``config`` is a dictionary containing algorithm-specific hyperparameters. To make sure we can call IRL algorithms in a unified way, you have to specify which hyperparameters your algorithm can take, as well as legal ranges and defaults. This is done as follows:

.. code-block:: python
    :emphasize-lines: 10-29
    
    from irl_benchmark.config import IRL_CONFIG_DOMAINS
    from irl_baselines.irl.algorithms.base_algorithm import BaseIRLAlgorithm

    class ExampleIRL(BaseIRLAlgorithm):
        """An example IRL algorithm"""
        # implementation here
        # ...
        # ...

    IRL_CONFIG_DOMAINS[ExampleIRL] = {
        'gamma': {
            'type': float,
            'min': 0.0,
            'max': 1.0,
            'default': 0.9,
        },
        'hyperparam1': {
            'type': 'categorical',
            'values': ['a', 'b'],
            'default': 'a',
        },
        'temperature': {
            'type': float,
            'optional': True,  # allows value to be None
            'min': 1e-10,
            'max': float('inf'),
            'default': None
        }
    }

Training
^^^^^^^^

The :class:`BaseIRLAlgorithm<.irl.algorithms.base_algorithm.BaseIRLAlgorithm>`: class provides the abstract method :meth:`train<.irl.algorithms.base_algorithm.BaseIRLAlgorithm.train>` as an interface of how IRL algorithms are run. You have to overwrite this method in your own implementation. The required parameters are:

* ``no_irl_iterations``: an integer specifying for how many iterations the algorithm should be run.
* ``no_rl_episodes_per_irl_iteration``: an integer specifying how many episodes the RL agent is allowed to run in each iteration.
* ``no_irl_episodes_per_irl_iteration``: an integer specifying how many episodes the IRL algorithm is allowed to run in addition to the RL episodes in each iteration. This can be used to collect empirical information with the trained agent, e.g. feature counts from the currently optimal policy. 

The train method returns a tuple containing the current reward function estimate on first position, and the trained agent on second position.

``TODO``: link here to description of the interface provided by the RL algorithm. Show code example

Useful methods
^^^^^^^^^^^^^^

The :class:`BaseIRLAlgorithm<.irl.algorithms.base_algorithm.BaseIRLAlgorithm>`: class comes with some useful methods that can be used in different subclasses. 

* There is a method to calculate discounted feature counts: :meth:`feature_count<.irl.algorithms.base_algorithm.BaseIRLAlgorithm.feature_count>`

RL algorithms
-------------

Metrics
-------




