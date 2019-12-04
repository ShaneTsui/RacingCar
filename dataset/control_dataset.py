import pickle
import random

from base.control import Control


class ControlDataset:

    def __init__(self, batch_size=512):
        self.rel_targets = []  # [rel_Xs | rel_Ys]
        self.controls = []  # [(steer, accel, break)]
        self.batch_size = batch_size
        self.idx = 0

    def load(self, save_path):
        with open(save_path, 'rb') as f:
            d = pickle.load(f)
            self.controls += d['controls']
            self.rel_targets += d['rel_targets']

    def save(self, save_path='../cache/control_dataset.pkl'):
        with open(save_path, 'wb') as f:
            pickle.dump({"controls": self.controls, "rel_targets": self.rel_targets}, f)

    def record(self, target, control: Control):
        self.controls.append(control)
        self.rel_targets.append(target)

    def size(self):
        return len(self.controls)

    def fetch_random_batch(self):
        self.rand_idx = [i for i in range(self.size())]
        idxes = random.choices(self.rand_idx, k=self.batch_size)

        x_batch = [self.rel_targets[i] for i in idxes]
        y_batch = [self.controls[i] for i in idxes]

        return x_batch, y_batch

    def __add__(self, other):
        dataset = ControlDataset()
        dataset.controls = self.controls + other.controls
        dataset.rel_targets = self.rel_targets + other.rel_targets
        return dataset


if __name__ == '__main__':
    dataset = ControlDataset()

    dataset.load('../cache/control_dataset.pkl')
    from utils.vis import plot_waypoints

    long_term, short_term = dataset.fetch_random_batch()

    for i in range(100):
        plot_waypoints(long_term[i], short_term[i])
