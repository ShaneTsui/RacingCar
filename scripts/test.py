from math import sin, cos

import gym
import torch

from base.environment import Environment
from nn.expert_policy import short_term_MPC
from nn.simple_policy import PolicyNetwork
from utils.vis import plot_waypoints


class Test:
    def __init__(self, env: Environment, model_dict):
        self.env = env
        self.dt = self.env.dt
        self.long_term_planning_length = self.env.long_term_planning_length
        self.short_term_planning_length = self.env.short_term_planning_length
        self.policy_layer = PolicyNetwork(self.long_term_planning_length * 2,
                                          self.short_term_planning_length * 2).cuda()
        self.policy_layer.load_state_dict(torch.load(model_dict))
        self.execution_layer = short_term_MPC

    def _to_absolute_coords(self, xs, ys, combine=True):
        car_x, car_y, _, _, theta = self.env.observe()

        xs_coords, ys_coords = [], []

        def _to_absolute_coord(x, y):
            x_ = cos(-theta) * x + sin(-theta) * y
            y_ = - sin(-theta) * x + cos(-theta) * y

            xs_coords.append(x_ + car_x)
            ys_coords.append(y_ + car_y)

        for x, y in zip(xs, ys):
            _to_absolute_coord(x, y)

        if not combine:
            return xs_coords, ys_coords
        return xs_coords + ys_coords

    def _to_relative_coords(self, xs, ys):
        car_x, car_y, _, _, theta = self.env.observe()

        xs_coords, ys_coords = [], []

        def _to_relative_coord(x, y):
            x -= car_x
            y -= car_y

            xs_coords.append(cos(theta) * x + sin(theta) * y)
            ys_coords.append(- sin(theta) * x + cos(theta) * y)

        for x, y in zip(xs, ys):
            _to_relative_coord(x, y)

        return xs_coords + ys_coords

    def run(self):
        self.env.reset()
        done = False
        with torch.no_grad():
            while not done:
                long_term_xs, long_term_ys = self.env.calc_long_term_targets()

                rel_long_term_waypoints = self._to_relative_coords(long_term_xs, long_term_ys)
                rel_nn_short_term_waypoints = self.policy_layer(
                    torch.FloatTensor(rel_long_term_waypoints).cuda()).detach().cpu().numpy()
                abs_nn_planned_xs, abs_nn_planned_ys = self._to_absolute_coords(
                    rel_nn_short_term_waypoints[:self.short_term_planning_length], \
                    rel_nn_short_term_waypoints[self.short_term_planning_length: 2 * self.short_term_planning_length],
                    combine=False)
                action = self.execution_layer(self.env.car, abs_nn_planned_xs, abs_nn_planned_ys, self.dt,
                                              self.short_term_planning_length)
                _, _, done, _ = self.env.step(action)
                plot_waypoints(rel_long_term_waypoints, rel_nn_short_term_waypoints)
                # self.env.render()


if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    env = Environment(env=env, FPS=50.0)
    test = Test(env=env, model_dict="saved/nn_policy_15.h5")
    test.run()
