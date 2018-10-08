import os

from slm_lab.agent import Agent
from slm_lab.env import OpenAIEnv
from slm_lab.experiment import analysis
from slm_lab.experiment.monitor import Body, InfoSpace, enable_aeb_space
from slm_lab.lib import logger, util
from slm_lab.spec import spec_util


class Session:
    '''The class which initializes the agent, environment, and runs them.'''

    def __init__(self, spec, info_space):
        self.spec = spec
        self.info_space = info_space
        self.index = self.info_space.get('session')

        # init singleton agent and env
        self.env = OpenAIEnv(self.spec)
        body = Body(self.env, self.spec['agent'])
        self.agent = Agent(self.spec, self.info_space, body=body)

        enable_aeb_space(self)  # to use lab's data analysis framework
        logger.info('Initialized session ' + str(self.index))

    def run_episode(self):
        self.env.clock.tick('epi')
        reward, state, done = self.env.reset()
        self.agent.reset(state)
        while not done:
            self.env.clock.tick('t')
            action = self.agent.act(state)
            reward, state, done = self.env.step(action)
            self.agent.update(action, reward, state, done)
        self.agent.body.log_summary()

    def close(self):
        self.agent.close()
        self.env.close()
        logger.info('Session done and closed.')

    def run(self):
        while self.env.clock.get('epi') <= self.env.max_episode:
            self.run_episode()
        self.data = analysis.analyze_session(self)  # session fitness
        self.close()
        return self.data


# To use SLM-Lab's existing spec. Alternatively, you can write one yourself too
spec = spec_util.get(spec_file='frozen.json',
                     spec_name='ddqn_epsilon_greedy_frozen')

# spec['env'][0]['name'] = 'LunarLander-v2'
# spec['env'][0]['name'] = 'FrozenLake-v0'
# spec['agent'][0]['algorithm']['explore_var_start'] = 1.0
# spec['agent'][0]['algorithm']['explore_var_end'] = 0.05
# spec['agent'][0]['algorithm']['explore_anneal_epi'] = 5000
# spec['agent'][0]['net']['hid_layers'] = []


info_space = InfoSpace()
# set proper env variables for the lab
os.environ['PREPATH'] = util.get_prepath(spec, info_space)
os.environ['lab_mode'] = 'train'  # set to 'train' to run at full speed

# inspect the agent spec; edit if you wish to
print(spec['agent'])
# edit the env spec to run for less episodes
spec['env'][0]['max_episode'] = 1000

# initialize and run session
sess = Session(spec, info_space)
data = sess.run()

print('Data is available at ' + str(util.smart_path(os.environ["PREPATH"])))
