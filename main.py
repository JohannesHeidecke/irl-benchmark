from itertools import product
import pandas as pd
import time
import os
import pickle


# FIXME, replace this with import:
def run(*args):
    return {'ile': 123, 'l2_loss': 234, 'avg_return': 345}


def main():

    env_ids = ['FrozenLake-v0', 'FrozenLake8x8-v0']
    agent_ids = [
        'ApprIRL-SVM', 'ApprIRL-Proj', 'MaxEntIRL', 'RelEntIRL', 'MaxCausIRL'
    ]
    no_expert_trajss = [10, 100, 1000, 10000]

    time_id = time.strftime('%Y-%m-%d_%H:%M:%S')
    dir_name = 'data/' + time_id
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    data = {}
    N = 5
    for i in range(N):
        for env_id, agent_id, no_expert_trajs in product(
                env_ids, agent_ids, no_expert_trajss):

            res = run(env_id, agent_id, no_expert_trajs)

            # Pickle the entire run result
            path = 'data/{}/{}_{}_{}_{}.pickle'.format(
                time_id, env_id, agent_id, no_expert_trajs, i)
            with open(path, 'wb') as f:
                pickle.dump(res, f)

            data.setdefault('env_id', []).append(env_id)
            data.setdefault('agent_id', []).append(agent_id)
            data.setdefault('no_expert_trajs', []).append(no_expert_trajs)

            data.setdefault('ile', []).append(res['ile'])
            data.setdefault('l2_loss', []).append(res['l2_loss'])
            data.setdefault('avg_return', []).append(res['avg_return'])

            data.setdefault('time_id', []).append(time_id)
            data.setdefault('run_number', []).append(i)

    df = pd.DataFrame(data)
    path = 'data/{}.csv'.format(time_id)
    df.to_csv(path)


if __name__ == '__main__':
    main()
