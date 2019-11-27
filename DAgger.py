from math import pi

from core.dataset import Dataset
from core.environment import Environment
from policy.expert_policy import short_term_MPC
from policy.simple_policy import PolicyNetwork


import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

N = 30
INF = float('inf')
MAX_steer = pi
MIN_steer = -pi
MAX_a = 10
MIN_a = -10
MAX_v = 100
MIN_v = 0


class DAgger:

    def __init__(self, env: Environment, state_dim, action_dim, batch_size, expert, lr):
        self.env = env
        self.expert = expert
        self.dt = self.env.dt
        self.long_term_planning_length = self.env.long_term_planning_length
        self.execution_layer = short_term_MPC

        self.policy_layer = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)


        self.batch_size = batch_size
        self.init_dataset_size = 20000
        self.dataset_prime_size = 10000
        self.M = 10
        self.waypoints_threshhold = 10 # TODO: Hyperparam

    def init_dataset(self):
        dataset = Dataset()
        while dataset.size() < self.init_dataset_size:
            done = False
            self.env.reset()
            while not done and dataset.size() < self.init_dataset_size: # TODO: Check if the env is really "done"
                long_term_targets = self.env.calc_long_term_targets()
                action, planned_waypoints = self.expert(self.env.car, long_term_targets, self.dt, self.long_term_planning_length)
                dataset.record(long_term_targets, planned_waypoints)
                _, _, done, _ = self.env.step(action)
        return dataset

    def measure_waypoints_diff(self, pnts1, pnts2):
        raise NotImplementedError()

    def train(self, dataset: Dataset):
        for inputs, outputs in dataset:   # TODO: implement this function
            outputs_pred = self.policy_layer(inputs)
            loss = F.mse_loss(outputs, outputs_pred)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def run_dagger(self):
        # Build D
        dataset = self.init_dataset()

        while self.policy_layer.is_converged(): # TODO: Implement
            # Train Policy NN with D

            self.train(dataset)

            # Run Policy NN + label with LT-MPC
            dataset_prime = Dataset()
            step = 0
            done = False
            while not done and dataset_prime.size() < self.dataset_prime_size:
                self.env.reset()
                while not done and dataset_prime.size() < self.dataset_prime_size:
                    long_term_targets = self.env.calc_long_term_targets()
                    nn_planned_waypoints = self.policy_layer(nn_planned_waypoints) # TODO: reshape the tensor

                    if not step % self.M:
                        _, expert_planned_waypoints = self.expert(self.env.car, long_term_targets, self.dt,
                                                            self.long_term_planning_length)
                        # TODO: Check if short-term conversion is need
                        if self.measure_waypoints_diff(nn_planned_waypoints, expert_planned_waypoints) > self.waypoints_threshhold:
                            dataset_prime.record(long_term_targets, expert_planned_waypoints)

                    action, _ = self.execution_layer(nn_planned_waypoints)  # TODO: nn VS expert?
                    _, _, done, _ = self.env.step(action)
                    step += 1

            # add D' to D
            dataset += dataset_prime

        self.save()

    def save(self, save_path='./nn_policy.h5'):
        self.policy_layer.save(save_path)

    def load(self):
        pass

    def policy(self):
        pass