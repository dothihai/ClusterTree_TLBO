import numpy as np
class Learner(object):
    """docstring for Learner"""
    def __init__(self, subjects):
        super(Learner, self).__init__()

        self.subjects = np.clip(subjects,0,0.99999999)
        self.seq = []
        self.tree = None
        self.fitness = 0
        # self.fitness,self.tree = fitness(subjects,node)