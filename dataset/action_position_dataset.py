import pickle
import random

from typing import Tuple


class ActionPositionDataset:

    def __init__(self, batch_size=512):
        self.control_trajs = []  # [as], a=(steer, accel, break)
        self.states = []  # [(vx, vy)]
        self.rel_position = []  # [rel_Xs | rel_Ys]
        self.batch_size = batch_size
        self.idx = 0

    def load(self, save_path):
        with open(save_path, 'rb') as f:
            d = pickle.load(f)
            self.control_trajs += d['control_trajs']
            self.states += d['states']
            self.rel_position += d['rel_position']

    def save(self, save_path='../cache/action_position_dataset.pkl'):
        with open(save_path, 'wb') as f:
            pickle.dump({"control_trajs": self.control_trajs, "states": self.states,
                         "rel_position": self.rel_position}, f)

    def record(self, actions: Tuple, state, positions):
        self.control_trajs.append(actions)
        self.states.append(state)
        self.rel_position.append(positions)

    def size(self):
        return len(self.control_trajs)

    def fetch_random_batch(self):
        self.rand_idx = [i for i in range(self.size())]
        idxes = random.choices(self.rand_idx, k=self.batch_size)

        x_batch = [self.control_trajs[i] + self.states[i] for i in idxes]
        y_batch = [self.rel_position[i] for i in idxes]

        return x_batch, y_batch

    def __add__(self, other):
        dataset = ActionPositionDataset()
        dataset.control_trajs = self.control_trajs + other.control_trajs
        dataset.states = self.states + other.states
        dataset.rel_position = self.rel_position + other.rel_position
        return dataset
