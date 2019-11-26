from math import pi

from core.dataset import Dataset
from core.environment import Environment
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

    def __init__(self, env: Environment, state_dim, action_dim, batch_size, expert):
        self.env = env
        self.expert = expert
        self.dt = self.env.dt
        self.policy_nn = PolicyNetwork(state_dim, action_dim)

        self.batch_size = batch_size
        self.dataset_size = 20000

    def train(self):
        # Build D
        # Train Policy NN with D

        # Run Policy NN + label with LT-MPC

        # compare and add D' to D
        self.save()

    def save(self):
        pass

    def load(self):
        pass

    def policy(self):
        pass