import random
from collections import deque

import numpy as np

from agent.memory.structure import SumTree


class Memory(object):

    def __init__(self, capacity):
        self.capacity = capacity

    def store(self, sample):
        pass

    def clear(self):
        pass

    def sample(self, batch_size):
        pass

    def update(self, *args, **kwargs):
        pass


class QueueMemory(Memory):

    def __init__(self, capacity):
        super(QueueMemory, self).__init__(capacity)
        self.queue = deque(maxlen=capacity)

    def store(self, sample):
        self.queue.append(sample)

    def clear(self):
        self.queue.clear()

    def sample(self, batch_size):
        return {'samples': random.sample(self.queue, batch_size)}


class PriorityMemory(Memory):

    def __init__(self, capacity, epsilon=0.01, alpha=0.6, beta=0.4, beta_increase=0.001, abs_err_upper=1):
        super(PriorityMemory, self).__init__(capacity)
        self.tree = SumTree(capacity)
        self.abs_err_upper = abs_err_upper
        self.beta = beta  # importance-sampling, from initial value increasing to 1
        self.beta_increment_per_sampling = beta_increase
        self.epsilon = epsilon  # small amount to avoid zero priority
        self.alpha = alpha  # [0~1] convert the importance of TD error to priority

    def store(self, sample):
        max_p = np.max(self.tree.get_data_priorities())
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, sample)

    def sample(self, batch_size):
        batch_samples = []
        tree_idx = []
        samples_weight = []
        segment = self.tree.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_p = np.min(self.tree.get_data_priorities() / self.tree.total())
        if min_p == 0:
            min_p = 0.00001
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, sample) = self.tree.get(s)
            prob = p / self.tree.total()
            samples_weight.append(np.power(prob / min_p, -self.beta))
            tree_idx.append(idx)
            batch_samples.append(sample)

        return {'samples': batch_samples, 'tree_idx': tree_idx, 'samples_weight': samples_weight}

    def update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for idx, p in zip(tree_idx, ps):
            self.tree.update(idx, p)

    def clear(self):
        self.tree.clear()


class EpisodeMemory(Memory):

    def __init__(self):
        super(EpisodeMemory, self).__init__(np.Inf)
        self.memory = []

    def store(self, sample):
        self.memory.append(sample)

    def sample(self, batch_size=np.Inf):
        return {'samples': self.memory}

    def clear(self):
        self.memory.clear()

#
# class PPOMemory(Memory):
#
#     def __init__(self):
#         super(PPOMemory, self).__init__(np.Inf)
#         self.memory = []
#
#     def store(self, sample):
#         self.memory.append(sample)
#
#     def sample(self, batch_size):
#         n = len(self.memory)
#         batch_starts = np.arange(0, n, batch_size)
#         indices = np.arange(n, dtype=np.int64)
#         np.random.shuffle(indices)
#         batches = [indices[i:i + batch_size] for i in batch_starts]
#         return {'samples': [self.memory[batch] for batch in batches]}
#
#     def clear(self):
#         self.memory.clear()


# class A2CMemory(Memory):
#
#     def __init__(self):
#         super(A2CMemory, self).__init__(np.Inf)
#         self.memory = []
#
#     def store(self, sample):
#         self.memory.append(sample)
#
#     def sample(self, batch_size):
#         return {'samples': self.memory}
#
#     def clear(self):
#         self.memory.clear()