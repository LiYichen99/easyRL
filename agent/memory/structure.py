import numpy as np


class SumTree(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1)
        self._data = np.zeros(capacity, dtype=object)
        self._data_point = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self._tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self._tree):
            return idx

        if s <= self._tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self._tree[left])

    def total(self):
        return self._tree[0]

    def update(self, idx, p):
        change = p - self._tree[idx]
        self._tree[idx] = p
        self._propagate(idx, change)

    def add(self, p, data):
        idx = self._data_point + self.capacity - 1

        self._data[self._data_point] = data
        self.update(idx, p)

        self._data_point += 1

        if self._data_point >= self.capacity:
            self._data_point = 0

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self._tree[idx], self._data[data_idx]

    def clear(self):
        self.__init__(capacity=self.capacity)

    def get_data_priorities(self):
        return self._tree[-self.capacity:]
