import pickle
import random

from base.action import Action


class WaypointPredictDataset:

    def __init__(self, batch_size=512):
        self.actions = []  # [(steer, accel, break)]
        self.real_rel_waypoints = []  # [rel_Xs | rel_Ys]
        self.batch_size = batch_size
        self.idx = 0

    def load(self, save_path):
        with open(save_path, 'rb') as f:
            d = pickle.load(f)
            self.actions += d['actions']
            self.rel_targets += d['rel_targets']
            self.rel_long_term_waypoints += d['rel_long_term_waypoints']

    def save(self, save_path='./action_dataset.pkl'):
        with open(save_path, 'wb') as f:
            pickle.dump({"actions": self.actions, "rel_targets": self.rel_targets,
                         "rel_long_term_waypoints": self.rel_long_term_waypoints}, f)

    def record(self, action: Action, target, long_term_waypoint):
        self.actions.append(action)
        self.rel_targets.append(target)
        self.rel_long_term_waypoints.append(long_term_waypoint)

    def size(self):
        return len(self.actions)

    def fetch_random_batch(self):
        self.rand_idx = [i for i in range(self.size())]
        idxes = random.choices(self.rand_idx, k=self.batch_size)

        x_batch = [self.rel_targets[i] for i in idxes]
        y_batch = [self.actions[i] for i in idxes]

        return x_batch, y_batch

    def __add__(self, other):
        dataset = WaypointPredictDataset()
        dataset.actions = self.actions + other.actions
        dataset.rel_targets = self.rel_targets + other.rel_targets
        dataset.rel_long_term_waypoints = self.rel_long_term_waypoints + other.rel_long_term_waypoints
        return dataset


if __name__ == '__main__':
    dataset = WaypointPredictDataset()

    dataset.load('../action_dataset.pkl')
    from utils.vis import plot_waypoints

    long_term, short_term = dataset.fetch_random_batch()

    for i in range(100):
        plot_waypoints(long_term[i], short_term[i])
