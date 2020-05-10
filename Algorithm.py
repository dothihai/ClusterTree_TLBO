class Learner(object):
    """docstring for Learner"""
    def __init__(self, subjects,node):
        super(Learner, self).__init__()

        self.subjects = subjects
        self.node = node
        self.fitness = 0
        # self.fitness,self.tree = fitness(subjects,node)