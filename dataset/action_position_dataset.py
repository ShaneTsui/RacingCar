import pickle
import random

from base.action import Action
from typing import List


class ActionPositionDataset:

    def __init__(self, batch_size=512):
        self.action_trajs = []  # [as], a=(steer, accel, break)
        self.states = []  # [(vx, vy)]
        self.rel_position = []  # [rel_Xs | rel_Ys]
        self.batch_size = batch_size
        self.idx = 0

    def load(self, save_path):
        with open(save_path, 'rb') as f:
            d = pickle.load(f)
            self.action_trajs += d['action_trajs']
            self.states += d['states']
            self.rel_position += d['rel_position']

    def save(self, save_path='../cache/action_position_dataset.pkl'):
        with open(save_path, 'wb') as f:
            pickle.dump({"action_trajs": self.action_trajs, "states": self.states,
                         "rel_position": self.rel_position}, f)

    def record(self, actions: List[Action], state, positions):
        self.action_trajs.append(actions)
        self.states.append(state)
        self.rel_position.append(positions)

    def size(self):
        return len(self.action_trajs)

    def fetch_random_batch(self):
        self.rand_idx = [i for i in range(self.size())]
        idxes = random.choices(self.rand_idx, k=self.batch_size)

        x_batch = [sum(a.to_tuple() for a in self.action_trajs[i]) + self.states[i] for i in idxes]
        y_batch = [self.rel_position[i] for i in idxes]

        return x_batch, y_batch

    def __add__(self, other):
        dataset = ActionPositionDataset()
        dataset.action_trajs = self.action_trajs + other.action_trajs
        dataset.states = self.states + other.states
        dataset.rel_position = self.rel_position + other.rel_position
        return dataset

