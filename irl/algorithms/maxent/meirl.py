from irl.base_algorithm import BaseIRLAlgorithm

class MaxEnt(BaseIRLAlgorithm):
    
    def __init__(self, env):
        super(MaxEnt, self).__init__(env)