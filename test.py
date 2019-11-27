import torch

from core.environment import Environment
from policy.simple_policy import PolicyNetwork


class Test:
    def __init__(self, env: Environment, model_dict):
        self.env = env
        self.long_term_planning_length = self.env.long_term_planning_length
        self.short_term_planning_length = self.env.short_term_planning_length
        self.policy_layer = PolicyNetwork(self.long_term_planning_length * 2,
                                          self.short_term_planning_length * 2).cuda()
        self.policy_layer.load_state_dict(torch.load(model_dict))

    def run(self):
        self.env.reset()
        while not done:
            self.env.render()
            long_term_xs, long_term_ys = self.env.calc_long_term_targets()

            rel_long_term_waypoints = self._to_relative_coords(long_term_xs, long_term_ys)
            rel_nn_planned_waypoints = self.policy_layer(torch.FloatTensor(rel_long_term_waypoints).cuda())
            rel_nn_planned_waypoints = rel_nn_planned_waypoints.detach().cpu().numpy()
            abs_nn_planned_xs, abs_nn_planned_ys = self._to_absolute_coords(
                rel_nn_planned_waypoints[:self.short_term_planning_length], \
                rel_nn_planned_waypoints[self.short_term_planning_length: 2 * self.short_term_planning_length],
                combine=False)

            action = self.execution_layer(self.env.car, abs_nn_planned_xs, abs_nn_planned_ys, self.dt,
                                          self.short_term_planning_length)

            _, _, done = self.env.step(action)