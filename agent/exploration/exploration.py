

class Exploration(object):

    def get_epsilon(self):
        pass

    def step(self):
        pass


class LinearExploration(Exploration):

    def __init__(self, init_epsilon, min_epsilon, epsilon_decay):
        self.epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay

    def get_epsilon(self):
        return self.epsilon

    def step(self):
        if self.epsilon * self.epsilon_decay > self.min_epsilon:
            self.epsilon = self.epsilon_decay * self.epsilon


class ConstExploration(Exploration):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self):
        return self.epsilon
