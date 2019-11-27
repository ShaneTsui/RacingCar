from math import pi
from math import sin, cos, sqrt

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from core.dataset import Dataset
from core.environment import Environment
from policy.expert_policy import short_term_MPC, long_term_MPC
from policy.simple_policy import PolicyNetwork

N = 30
INF = float('inf')
MAX_steer = pi
MIN_steer = -pi
MAX_a = 10
MIN_a = -10
MAX_v = 100
MIN_v = 0

class DAgger:

    def __init__(self, env: Environment, lr=0.01):
        self.env = env
        self.expert = long_term_MPC
        self.dt = self.env.dt
        self.long_term_planning_length = self.env.long_term_planning_length
        self.short_term_planning_length = self.env.short_term_planning_length
        self.execution_layer = short_term_MPC

        self.policy_layer = PolicyNetwork(self.long_term_planning_length * 2,
                                          self.short_term_planning_length * 2).cuda()
        self.optimizer = optim.Adam(self.policy_layer.parameters(), lr=lr)

        self.init_dataset_size = 20000
        self.dataset_prime_size = 1000
        self.M = 5
        self.waypoints_threshhold = 1

    def _to_absolute_coords(self, xs, ys, combine=True):
        car_x, car_y, _, _, theta = self.env.observe()

        xs_coords, ys_coords = [], []

        def to_relative_coord(x, y):
            x_ = cos(-theta) * x + sin(-theta) * y
            y_ = - sin(-theta) * x + cos(-theta) * y

            xs_coords.append(x_ + car_x)
            ys_coords.append(y_ + car_y)

        for x, y in zip(xs, ys):
            to_relative_coord(x, y)

        if not combine:
            return xs_coords, ys_coords
        return xs_coords + ys_coords

    def _to_relative_coords(self, xs, ys):
        car_x, car_y, _, _, theta = self.env.observe()

        xs_coords, ys_coords = [], []

        def to_relative_coord(x, y):
            x -= car_x
            y -= car_y

            xs_coords.append(cos(theta) * x + sin(theta) * y)
            ys_coords.append(- sin(theta) * x + cos(theta) * y)

        for x, y in zip(xs, ys):
            to_relative_coord(x, y)

        return xs_coords + ys_coords

    def init_dataset(self, save=True):
        dataset = Dataset()
        while dataset.size() < self.init_dataset_size:
            done = False
            self.env.reset()
            while not done and dataset.size() < self.init_dataset_size:  # TODO: Check if the env is really "done"
                long_term_xs, long_term_ys = self.env.calc_long_term_targets()
                action, expert_xs, expert_ys = self.expert(self.env.car, long_term_xs, long_term_ys, self.dt,
                                                           self.long_term_planning_length)
                rel_long_term_waypoints = self._to_relative_coords(long_term_xs, long_term_ys)
                rel_planned_waypoints = self._to_relative_coords(expert_xs[:self.short_term_planning_length],
                                                                 expert_ys[:self.short_term_planning_length])
                dataset.record(rel_long_term_waypoints, rel_planned_waypoints)
                _, _, done = self.env.step(action)
        if save:
            dataset.save()
        return dataset

    def _measure_waypoints_diff(self, pnts1, pnts2):
        return sqrt(sum([(a - b) ** 2 for a, b in zip(pnts1, pnts2)]))

    def _train(self, dataset: Dataset):
        while True:
            batch = dataset.get_batch()
            if not batch:
                break
            inputs, outputs = batch
            inputs = torch.FloatTensor(inputs).cuda()
            outputs = torch.FloatTensor(outputs).cuda()

            outputs_pred = self.policy_layer(inputs)
            loss = F.mse_loss(outputs, outputs_pred)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run_dagger(self):
        # Build D
        dataset = self.init_dataset()
        print("Done init")

        # while self.policy_layer.is_converged():  # TODO: Implement
        while True:
            # Train Policy NN with D
            self._train(dataset)
            round = 0
            # Run Policy NN + label with LT-MPC
            dataset_prime = Dataset()
            step = 0
            done = False
            while not done and dataset_prime.size() < self.dataset_prime_size:
                self.env.reset()
                while not done and dataset_prime.size() < self.dataset_prime_size:
                    long_term_xs, long_term_ys = self.env.calc_long_term_targets()

                    rel_long_term_waypoints = self._to_relative_coords(long_term_xs, long_term_ys)
                    rel_nn_planned_waypoints = self.policy_layer(torch.FloatTensor(rel_long_term_waypoints).cuda())
                    rel_nn_planned_waypoints = rel_nn_planned_waypoints.detach().cpu().numpy()

                    if not step % self.M:
                        _, expert_xs, expert_ys = self.expert(self.env.car, long_term_xs, long_term_ys, self.dt,
                                                              self.long_term_planning_length)
                        rel_expert_planned_waypoints = self._to_relative_coords(
                            expert_xs[:self.short_term_planning_length], expert_ys[:self.short_term_planning_length])
                        diff = self._measure_waypoints_diff(rel_nn_planned_waypoints,
                                                            rel_expert_planned_waypoints)
                        print(diff)
                        if diff > self.waypoints_threshhold:
                            dataset_prime.record(rel_long_term_waypoints, rel_expert_planned_waypoints)

                    abs_nn_planned_xs, abs_nn_planned_ys = self._to_absolute_coords(
                        rel_nn_planned_waypoints[:self.short_term_planning_length], \
                        rel_nn_planned_waypoints[self.short_term_planning_length: 2 * self.short_term_planning_length],
                        combine=False)
                    action = self.execution_layer(self.env.car, abs_nn_planned_xs, abs_nn_planned_ys, self.dt,
                                                  self.short_term_planning_length)
                    _, _, done = self.env.step(action)
                    step += 1
            round += 1
            self.save("./saved/nn_policy_{}.h5".format(round))

            # add D' to D
            dataset += dataset_prime


    def save(self, save_path='./nn_policy.h5'):
        torch.save(self.policy_layer.state_dict(), save_path)

    def load(self):
        pass

    def policy(self):
        pass


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env = Environment(env=env, FPS=50.0)

    dagger = DAgger(env)
    dagger.run_dagger()
