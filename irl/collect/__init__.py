import pickle
import os

from irl.feature.feature_wrapper import FeatureWrapper

def collect_trajs(env, agent, no_episodes, max_steps_per_episode, store_to=None):

    trajectories = []

    for episode in range(no_episodes):
        state = env.reset()
        done = False

        states = [state]
        actions = []
        rewards = []
        true_rewards = []
        features = []

        step_counter = 0
        while not done and step_counter < max_steps_per_episode:
            step_counter += 1
            action = agent.pick_action(state)
            next_state, reward, done, info = env.step(action)
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            if 'true_reward' in info.keys():
                true_rewards.append(info['true_reward'])
            if 'features' in info.keys():
                features.append(info['features'])
            state = next_state

        trajectory = {
            'states': states, 
            'actions': actions,
            'rewards': rewards,
            'true_rewards': true_rewards,
            'features': features
        }
        trajectories.append(trajectory)

    if store_to is not None:
        if not os.path.exists(store_to):
            os.makedirs(store_to)
        with open(store_to + 'trajs.pkl', 'wb+') as f:
            pickle.dump(trajectories, f)
    
    return trajectories
    



