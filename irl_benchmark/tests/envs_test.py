import gym

from irl_benchmark.envs import envs_feature_based, envs_known_transitions, env_ids
from irl_benchmark.irl.feature import feature_wrapper
from irl_benchmark.utils.utils import get_transition_matrix


def test_feature_wrapped_envs():
    '''Tests the set of feature-wrappable environments.'''
    feature_wrappable_envs = envs_feature_based()
    assert isinstance(feature_wrappable_envs, set)
    assert len(feature_wrappable_envs) > 0
    assert len(feature_wrappable_envs) <= len(env_ids)
    for env_id in feature_wrappable_envs:
        feature_wrapper.make(env_id)

def test_transitions_provided():
    '''Tests the set of environments with known tranistion dynamics.'''
    transitions_provided_ids = envs_known_transitions()
    assert isinstance(transitions_provided_ids, set)
    assert len(transitions_provided_ids) > 0
    assert len(transitions_provided_ids) <= len(env_ids)
    for env_id in transitions_provided_ids:
        get_transition_matrix(gym.make(env_id))
